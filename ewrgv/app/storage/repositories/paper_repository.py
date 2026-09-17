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

    # ------------------------------------------------------------------ #
    # Phase 2.5 — Document Parsing
    # ------------------------------------------------------------------ #

    async def get_papers_for_parsing(self, run_id: str) -> list[dict]:
        """
        Fetches all papers linked to a specific research run that are
        eligible for document parsing.

        Eligible papers must:
        - Have full_text_status = 'DOWNLOADED'  (PDF has been acquired)
        - Have parsing_status != 'PARSED'        (not already parsed)

        This makes the operation idempotent: already-PARSED papers are
        naturally skipped, and PARSE_FAILED papers can be retried.
        """
        query = """
            SELECT DISTINCT
                p.paper_id,
                p.title,
                p.doi,
                p.full_text_status,
                p.storage_path,
                p.parsing_status,
                p.parsed_storage_path,
                p.page_count
            FROM papers p
            JOIN paper_sources ps ON ps.paper_id = p.paper_id
            WHERE ps.run_id = $1
              AND p.full_text_status = 'DOWNLOADED'
              AND COALESCE(p.parsing_status, 'NOT_PARSED') != 'PARSED'
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, run_id)
            return [dict(r) for r in rows]

    async def update_parsing_metadata(
        self,
        paper_id: str,
        status: str,
        parsed_storage_path: str | None = None,
        page_count: int | None = None,
    ) -> None:
        """
        Updates the document parsing metadata for a specific paper.

        Parameters
        ----------
        paper_id:
            UUID of the paper to update.
        status:
            New parsing_status value.  Expected values:
            'PARSED', 'PARSE_FAILED', 'INVALID_DOCUMENT', 'NOT_PARSED'.
        parsed_storage_path:
            Bucket-relative path to the uploaded parsed.json file,
            or None if parsing was unsuccessful.
        page_count:
            Number of pages extracted, or None if parsing failed.
        """
        query = """
            UPDATE papers
            SET parsing_status      = $1,
                parsed_storage_path = $2,
                page_count          = $3,
                parsed_at           = CASE WHEN $1 = 'PARSED' THEN CURRENT_TIMESTAMP ELSE parsed_at END,
                updated_at          = CURRENT_TIMESTAMP
            WHERE paper_id = $4
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                query,
                status,
                parsed_storage_path,
                page_count,
                paper_id,
            )

    # ------------------------------------------------------------------ #
    # Phase 2.6 -- Section Extraction
    # ------------------------------------------------------------------ #

    async def get_papers_for_sectioning(self, run_id: str) -> list[dict]:
        """
        Fetches all papers eligible for section extraction.

        Eligible papers must:
        - Have full_text_status = 'DOWNLOADED'
        - Have parsing_status   = 'PARSED'
        - Have section_status  != 'EXTRACTED'
        """
        query = """
            SELECT DISTINCT
                p.paper_id,
                p.title,
                p.doi,
                p.full_text_status,
                p.parsing_status,
                p.parsed_storage_path,
                p.section_status,
                p.sections_storage_path
            FROM papers p
            JOIN paper_sources ps ON ps.paper_id = p.paper_id
            WHERE ps.run_id = $1
              AND p.full_text_status = 'DOWNLOADED'
              AND COALESCE(p.parsing_status, 'NOT_PARSED') = 'PARSED'
              AND COALESCE(p.section_status, 'NOT_EXTRACTED') != 'EXTRACTED'
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, run_id)
            return [dict(r) for r in rows]

    async def update_section_metadata(
        self,
        paper_id: str,
        status: str,
        sections_storage_path: str | None = None,
        sections_count: int | None = None,
    ) -> None:
        """Updates section extraction metadata for a specific paper."""
        query = """
            UPDATE papers
            SET section_status          = $1,
                sections_storage_path   = COALESCE($2, sections_storage_path),
                sections_count          = COALESCE($3, sections_count),
                sectioned_at            = CASE WHEN $1 = 'EXTRACTED' THEN CURRENT_TIMESTAMP ELSE sectioned_at END,
                updated_at              = CURRENT_TIMESTAMP
            WHERE paper_id = $4
        """
        async with self._pool.acquire() as conn:
            await conn.execute(query, status, sections_storage_path, sections_count, paper_id)

    # ------------------------------------------------------------------ #
    # Phase 2.7 -- Chunking
    # ------------------------------------------------------------------ #

    async def get_papers_for_chunking(self, run_id: str) -> list[dict]:
        """
        Fetches all papers eligible for chunking.

        Eligible papers must:
        - Have full_text_status = 'DOWNLOADED'
        - Have parsing_status   = 'PARSED'
        - Have section_status   = 'EXTRACTED'
        - Have chunk_status    != 'CHUNKED'
        """
        query = """
            SELECT DISTINCT
                p.paper_id,
                p.sections_storage_path,
                p.sections_count
            FROM papers p
            JOIN paper_sources ps ON ps.paper_id = p.paper_id
            WHERE ps.run_id = $1
              AND p.full_text_status = 'DOWNLOADED'
              AND p.parsing_status = 'PARSED'
              AND p.section_status = 'EXTRACTED'
              AND COALESCE(p.chunk_status, 'NOT_CHUNKED') != 'CHUNKED'
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, run_id)
            return [dict(r) for r in rows]

    async def update_chunk_metadata(
        self,
        paper_id: str,
        status: str,
        chunks_storage_path: str | None = None,
        chunks_count: int | None = None,
    ) -> None:
        """Updates chunking metadata for a specific paper."""
        query = """
            UPDATE papers
            SET chunk_status        = $1,
                chunks_storage_path = COALESCE($2, chunks_storage_path),
                chunks_count        = COALESCE($3, chunks_count),
                chunked_at          = CASE WHEN $1 = 'CHUNKED' THEN CURRENT_TIMESTAMP ELSE chunked_at END,
                updated_at          = CURRENT_TIMESTAMP
            WHERE paper_id = $4
        """
        async with self._pool.acquire() as conn:
            await conn.execute(query, status, chunks_storage_path, chunks_count, paper_id)

    # ------------------------------------------------------------------ #
    # Phase 2.9 -- Text Representation (Dense + Sparse)
    # ------------------------------------------------------------------ #

    async def get_papers_for_representation(self, run_id: str) -> list[dict]:
        """
        Fetches all papers eligible for representation processing.

        Eligible papers must:
        - Have chunk_status = 'CHUNKED'
        """
        query = """
            SELECT DISTINCT
                p.paper_id,
                p.chunks_storage_path,
                p.chunks_count
            FROM papers p
            JOIN paper_sources ps ON ps.paper_id = p.paper_id
            WHERE ps.run_id = $1
              AND p.chunk_status = 'CHUNKED'
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, run_id)
            return [dict(r) for r in rows]

    async def update_embedding_status(
        self,
        paper_id: str,
        status: str,
    ) -> None:
        """Updates embedding status metadata for a specific paper."""
        query = """
            UPDATE papers
            SET embedding_status = $1,
                embedded_at      = CASE WHEN $1 = 'EMBEDDED' THEN CURRENT_TIMESTAMP ELSE embedded_at END,
                updated_at       = CURRENT_TIMESTAMP
            WHERE paper_id = $2
        """
        async with self._pool.acquire() as conn:
            await conn.execute(query, status, paper_id)

    # ------------------------------------------------------------------ #
    # Phase 2.8 — Corpus Health & Consistency
    # ------------------------------------------------------------------ #

    async def get_corpus_health_counts(self, run_id: str | None = None) -> dict:
        """
        Return aggregate processing-status counts for the corpus.

        Parameters
        ----------
        run_id:
            If provided, restrict counts to papers linked to this research run.
            If None, counts span the entire corpus.

        Returns
        -------
        dict with keys:
            total_papers, downloaded, not_checked, no_oa_found,
            download_failed, invalid_document, parsed,
            sections_extracted, chunked
        """
        if run_id is not None:
            query = """
                SELECT
                    COUNT(DISTINCT p.paper_id)                                            AS total_papers,
                    COUNT(DISTINCT p.paper_id) FILTER (WHERE p.full_text_status = 'DOWNLOADED')       AS downloaded,
                    COUNT(DISTINCT p.paper_id) FILTER (WHERE p.full_text_status = 'NOT_CHECKED')      AS not_checked,
                    COUNT(DISTINCT p.paper_id) FILTER (WHERE p.full_text_status = 'NO_OA_FOUND')      AS no_oa_found,
                    COUNT(DISTINCT p.paper_id) FILTER (WHERE p.full_text_status = 'DOWNLOAD_FAILED')  AS download_failed,
                    COUNT(DISTINCT p.paper_id) FILTER (WHERE p.full_text_status = 'INVALID_DOCUMENT') AS invalid_document,
                    COUNT(DISTINCT p.paper_id) FILTER (WHERE p.parsing_status  = 'PARSED')           AS parsed,
                    COUNT(DISTINCT p.paper_id) FILTER (WHERE p.section_status  = 'EXTRACTED')        AS sections_extracted,
                    COUNT(DISTINCT p.paper_id) FILTER (WHERE p.chunk_status    = 'CHUNKED')          AS chunked
                FROM papers p
                JOIN paper_sources ps ON ps.paper_id = p.paper_id
                WHERE ps.run_id = $1
            """
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(query, run_id)
        else:
            query = """
                SELECT
                    COUNT(*)                                                              AS total_papers,
                    COUNT(*) FILTER (WHERE full_text_status = 'DOWNLOADED')              AS downloaded,
                    COUNT(*) FILTER (WHERE full_text_status = 'NOT_CHECKED')             AS not_checked,
                    COUNT(*) FILTER (WHERE full_text_status = 'NO_OA_FOUND')             AS no_oa_found,
                    COUNT(*) FILTER (WHERE full_text_status = 'DOWNLOAD_FAILED')         AS download_failed,
                    COUNT(*) FILTER (WHERE full_text_status = 'INVALID_DOCUMENT')        AS invalid_document,
                    COUNT(*) FILTER (WHERE parsing_status  = 'PARSED')                  AS parsed,
                    COUNT(*) FILTER (WHERE section_status  = 'EXTRACTED')               AS sections_extracted,
                    COUNT(*) FILTER (WHERE chunk_status    = 'CHUNKED')                 AS chunked
                FROM papers
            """
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(query)

        if row is None:
            return {}

        return {
            "total_papers": row["total_papers"] or 0,
            "downloaded": row["downloaded"] or 0,
            "not_checked": row["not_checked"] or 0,
            "no_oa_found": row["no_oa_found"] or 0,
            "download_failed": row["download_failed"] or 0,
            "invalid_document": row["invalid_document"] or 0,
            "parsed": row["parsed"] or 0,
            "sections_extracted": row["sections_extracted"] or 0,
            "chunked": row["chunked"] or 0,
        }

    async def get_papers_for_consistency_check(
        self,
        run_id: str | None = None,
    ) -> list[dict]:
        """
        Fetch all papers with their processing status fields for consistency checking.

        Returns enough information for CorpusHealthService to determine which
        artifacts should exist and which DB/storage pairs to check.

        Parameters
        ----------
        run_id:
            If provided, restrict to papers in that research run.
            If None, return all papers in the corpus.
        """
        if run_id is not None:
            query = """
                SELECT DISTINCT
                    p.paper_id,
                    p.full_text_status,
                    p.storage_path,
                    p.parsing_status,
                    p.parsed_storage_path,
                    p.section_status,
                    p.sections_storage_path,
                    p.chunk_status,
                    p.chunks_storage_path
                FROM papers p
                JOIN paper_sources ps ON ps.paper_id = p.paper_id
                WHERE ps.run_id = $1
                ORDER BY p.paper_id
            """
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query, run_id)
        else:
            query = """
                SELECT
                    paper_id,
                    full_text_status,
                    storage_path,
                    parsing_status,
                    parsed_storage_path,
                    section_status,
                    sections_storage_path,
                    chunk_status,
                    chunks_storage_path
                FROM papers
                ORDER BY paper_id
            """
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(query)

        return [dict(r) for r in rows]

    async def get_provenance_for_paper(self, paper_id: str) -> list[dict]:
        """
        Retrieve all provenance records (paper_sources rows) for a canonical paper.

        Useful for verifying that provenance is preserved across multiple runs
        and providers.

        Returns
        -------
        list[dict]:
            All paper_sources rows referencing this paper_id.
            Each dict contains: source_id, paper_id, run_id, provider,
            provider_paper_id, source_query, paper_url, citation_count, created_at.
        """
        query = """
            SELECT
                source_id,
                paper_id,
                run_id,
                provider,
                provider_paper_id,
                source_query,
                paper_url,
                citation_count,
                created_at
            FROM paper_sources
            WHERE paper_id = $1
            ORDER BY created_at ASC
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, paper_id)
        return [dict(r) for r in rows]

    async def get_paper_by_id(self, paper_id: str) -> dict | None:
        """
        Fetch a single canonical paper by paper_id.

        Returns None if not found.
        """
        query = """
            SELECT
                paper_id, title, abstract, doi, publication_year, authors,
                full_text_status, storage_path, parsing_status, parsed_storage_path,
                section_status, sections_storage_path, chunk_status, chunks_storage_path,
                created_at, updated_at
            FROM papers
            WHERE paper_id = $1
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, paper_id)
        return dict(row) if row else None
