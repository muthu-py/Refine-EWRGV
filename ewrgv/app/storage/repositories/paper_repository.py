"""
app/storage/repositories/paper_repository.py
--------------------------------------------
Repository for persisting collected papers to Supabase PostgreSQL.
"""

import json
from asyncpg import Pool
from app.core.logging import get_logger
from app.domain.models.collection import CollectionResult

logger = get_logger(__name__)


class PaperRepository:
    """Handles database operations for the papers and paper_sources tables."""

    def __init__(self, db_pool: Pool):
        self._pool = db_pool

    async def save_collection_result(self, result: CollectionResult, research_question: str) -> None:
        """
        Persists a CollectionResult into `research_runs`, `papers`, and `paper_sources` tables.
        Uses an upsert strategy for `papers` based on DOI if present.
        """
        if not result.papers:
            return

        query_runs = """
            INSERT INTO research_runs (run_id, research_question)
            VALUES ($1, $2)
            ON CONFLICT (run_id) DO NOTHING
        """

        query_papers = """
            INSERT INTO papers (title, abstract, doi, publication_year, authors, full_text_url, source_type)
            VALUES ($1, $2, $3, $4, $5::jsonb, $6, $7)
            ON CONFLICT (doi) DO UPDATE SET
                title = COALESCE(EXCLUDED.title, papers.title),
                abstract = COALESCE(papers.abstract, EXCLUDED.abstract),
                publication_year = COALESCE(papers.publication_year, EXCLUDED.publication_year),
                authors = EXCLUDED.authors,
                full_text_url = COALESCE(papers.full_text_url, EXCLUDED.full_text_url),
                source_type = COALESCE(papers.source_type, EXCLUDED.source_type),
                updated_at = CURRENT_TIMESTAMP
            RETURNING paper_id;
        """

        query_sources = """
            INSERT INTO paper_sources (paper_id, run_id, provider, provider_paper_id, source_query, paper_url, citation_count)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """

        saved_count = 0

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Persist the research run record
                await conn.execute(query_runs, result.query_id, research_question)

                for cp in result.papers:
                    paper = cp.paper
                    prov = cp.provenance

                    authors_json = json.dumps([a.model_dump() for a in paper.authors])
                    doi = paper.doi.strip() if paper.doi else None

                    # Upsert paper
                    row = await conn.fetchrow(
                        query_papers,
                        paper.title,
                        paper.abstract,
                        doi,
                        paper.publication_year,
                        authors_json,
                        paper.full_text_url,
                        prov.source if paper.full_text_url else None
                    )
                    
                    if row and row['paper_id']:
                        # Insert provenance source linking paper to run
                        await conn.execute(
                            query_sources,
                            row['paper_id'],
                            result.query_id,
                            prov.source,
                            prov.provider_id,
                            prov.source_query,
                            paper.paper_url,
                            paper.citation_count
                        )
                        saved_count += 1
                        
        logger.info(
            "PaperRepository.save_collection_result: completed",
            extra={
                "run_id": result.query_id,
                "saved_sources": saved_count
            }
        )

    async def get_papers_pending_acquisition(self, limit: int = 100) -> list[dict]:
        """
        Fetches papers that have not yet had full-text acquisition attempted.
        Returns a list of dicts representing the rows.
        """
        query = """
            SELECT paper_id, title, doi, full_text_status, full_text_url, source_type
            FROM papers
            WHERE full_text_status = 'NOT_CHECKED'
            ORDER BY created_at DESC
            LIMIT $1
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, limit)
            return [dict(r) for r in rows]

    async def get_papers_for_run(self, run_id: str) -> list[dict]:
        """
        Fetches all papers linked to a specific research run that have
        not yet been checked for full-text acquisition.

        Joining through paper_sources ensures we only process papers
        that were discovered in this run while still respecting the
        corpus-wide deduplication (canonical paper_id).
        """
        query = """
            SELECT DISTINCT p.paper_id, p.title, p.doi,
                   p.full_text_status, p.full_text_url, p.source_type
            FROM papers p
            JOIN paper_sources ps ON ps.paper_id = p.paper_id
            WHERE ps.run_id = $1
              AND p.full_text_status = 'NOT_CHECKED'
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, run_id)
            return [dict(r) for r in rows]

    async def run_exists(self, run_id: str) -> bool:
        """Returns True if a research_runs row exists for run_id."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchval(
                "SELECT 1 FROM research_runs WHERE run_id = $1", run_id
            )
            return row is not None

    async def update_fulltext_metadata(
        self,
        paper_id: str,
        status: str,
        full_text_url: str | None = None,
        storage_path: str | None = None,
        source_type: str | None = None,
        content_type: str | None = None,
        file_size: int | None = None,
    ) -> None:
        """
        Updates the full-text acquisition metadata for a specific paper.
        """
        query = """
            UPDATE papers
            SET full_text_status = $1,
                full_text_url = COALESCE($2, full_text_url),
                storage_path = $3,
                source_type = COALESCE($4, source_type),
                content_type = $5,
                file_size = $6,
                acquired_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE paper_id = $7
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                query,
                status,
                full_text_url,
                storage_path,
                source_type,
                content_type,
                file_size,
                paper_id,
            )
