"""
app/ingestion/representation/service.py
---------------------------------------
Orchestrates Phase 2.9 text representation logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.core.exceptions import EmbeddingAPIError, StorageError
from app.core.job_store import JobStatus, RepresentationJob
from app.core.logging import get_logger
from app.domain.enums import EmbeddingStatus
from app.domain.interfaces import EmbeddingProvider
from app.domain.models.paper_chunk import ChunkingResult, PaperChunk
from app.retrieval.bm25_index import BM25Index
from app.storage.artifact_paths import ARTIFACT_BUCKET, get_chunks_path
from app.storage.repositories.embedding_repository import (
    EmbeddingRecord,
    EmbeddingRepository,
)
from app.storage.repositories.paper_repository import PaperRepository
from app.storage.supabase_storage import SupabaseStorageClient

logger = get_logger(__name__)


class RepresentationService:
    """
    Service for building text representations (dense + sparse).
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        embedding_repo: EmbeddingRepository,
        paper_repo: PaperRepository,
        storage_client: SupabaseStorageClient,
    ) -> None:
        self._provider = embedding_provider
        self._embedding_repo = embedding_repo
        self._paper_repo = paper_repo
        self._storage_client = storage_client

    async def run_representation_for_run(
        self,
        run_id: str,
        job: RepresentationJob,
    ) -> None:
        """
        Execute representation building for all chunked papers in a run.
        Updates the RepresentationJob in-place.
        """
        # 1. Fetch eligible papers (status checks will be handled by the repo query)
        papers = await self._paper_repo.get_papers_for_representation(run_id)
        if not papers:
            logger.info("No eligible papers found for representation in run %s", run_id)
            job.status = JobStatus.COMPLETED
            job.embedding_status = "NOT_EMBEDDED"
            job.bm25_status = "NOT_INDEXED"
            return

        job.total_papers = len(papers)
        logger.info("Found %d papers for representation (run %s)", len(papers), run_id)

        all_run_chunks: list[PaperChunk] = []
        # Set when the embedding API fails in a way retrying won't fix this run
        # (bad key, rate limit after retries); remaining batches are skipped.
        stop_error: Optional[EmbeddingAPIError] = None

        # 2. Process each paper for embeddings
        for paper in papers:
            paper_id = str(paper["paper_id"])  # asyncpg returns uuid.UUID; chunk_embeddings.paper_id is TEXT
            try:
                # 2a. Download chunks.json
                chunks_path = get_chunks_path(paper_id)
                raw_bytes = await self._storage_client.download(ARTIFACT_BUCKET, chunks_path)
                data = json.loads(raw_bytes)
                chunk_result = ChunkingResult.model_validate(data)
                
                paper_chunks = chunk_result.chunks
                all_run_chunks.extend(paper_chunks)
                job.total_chunks += len(paper_chunks)

                if not paper_chunks:
                    continue

                # 2b. Check existing embeddings (Incremental)
                model_name = self._provider.get_model_name()
                existing_ids = await self._embedding_repo.get_embedded_chunk_ids(paper_id, model_name)

                # Re-chunking issues new chunk_ids; drop vectors of chunks that no
                # longer exist so counts and paper status stay accurate.
                stale_ids = existing_ids - {c.chunk_id for c in paper_chunks}
                if stale_ids:
                    await self._embedding_repo.delete_embeddings(sorted(stale_ids), model_name)
                    existing_ids = existing_ids - stale_ids
                    logger.info("Paper %s: pruned %d stale embeddings.", paper_id, len(stale_ids))
                
                # Empty chunks are skipped: providers reject empty inputs (Gemini: 400).
                embeddable = [c for c in paper_chunks if c.text and c.text.strip()]
                chunks_to_embed = [c for c in embeddable if c.chunk_id not in existing_ids]
                
                if not chunks_to_embed:
                    logger.debug("Paper %s completely embedded, skipping.", paper_id)
                    job.embedded_chunks += len(paper_chunks)
                    await self._update_paper_status(paper_id, EmbeddingStatus.EMBEDDED)
                    job.processed_papers += 1
                    continue

                logger.info(
                    "Paper %s: %d total chunks, %d already embedded, %d to process.",
                    paper_id, len(paper_chunks), len(existing_ids), len(chunks_to_embed)
                )

                # 2c. Batch embed
                batch_size = settings.EMBEDDING_BATCH_SIZE
                for i in range(0, len(chunks_to_embed), batch_size):
                    batch = chunks_to_embed[i : i + batch_size]
                    texts = [c.text for c in batch]

                    if stop_error is not None:
                        job.failed_chunks += len(batch)
                        continue

                    try:
                        embeddings = await self._provider.embed_documents(texts)
                        
                        records = []
                        for chunk, emb in zip(batch, embeddings):
                            records.append(
                                EmbeddingRecord(
                                    chunk_id=chunk.chunk_id,
                                    paper_id=paper_id,
                                    embedding=emb,
                                    embedding_model=model_name,
                                    embedding_dimension=self._provider.get_dimension(),
                                )
                            )
                            
                        # 2d. Save embeddings
                        await self._embedding_repo.save_embeddings(records)
                        job.embedded_chunks += len(records)
                        
                    except EmbeddingAPIError as exc:
                        logger.error("Failed to embed batch for paper %s: %s", paper_id, exc)
                        job.failed_chunks += len(batch)
                        job.error = exc.message
                        if exc.kind in ("auth", "rate_limit"):
                            stop_error = exc
                    except Exception as exc:
                        logger.error("Failed to embed batch for paper %s: %s", paper_id, exc)
                        job.failed_chunks += len(batch)
                        job.error = f"Embedding failed: {exc}"
                        # Continue with next batch/paper

                # Determine paper status
                total_expected = len(embeddable)
                actual_embedded = await self._embedding_repo.get_embedding_counts(paper_id, model_name)
                
                if actual_embedded >= total_expected:
                    status = EmbeddingStatus.EMBEDDED
                elif actual_embedded > 0:
                    status = EmbeddingStatus.PARTIAL
                else:
                    status = EmbeddingStatus.EMBEDDING_FAILED

                await self._update_paper_status(paper_id, status)
                job.processed_papers += 1
                
            except Exception as exc:
                logger.error("Failed to process paper %s for representation: %s", paper_id, exc)
                job.failed_papers += 1

        # 3. Build and persist BM25 index
        try:
            if all_run_chunks:
                bm25_index = BM25Index().build(all_run_chunks)
                index_dir = Path(settings.BM25_INDEX_PATH) / run_id
                index_path = index_dir / "bm25_index.pkl"
                bm25_index.save(index_path)
                job.bm25_status = "INDEXED"
            else:
                job.bm25_status = "NO_CHUNKS"
        except Exception as exc:
            logger.error("Failed to build BM25 index for run %s: %s", run_id, exc)
            job.bm25_status = "FAILED"
            job.error = "; ".join(filter(None, [job.error, f"BM25 build failed: {str(exc)}"]))

        if stop_error is not None:
            job.error = (
                f"{stop_error.message}. Embedding stopped early; re-run /represent "
                "to embed only the remaining chunks."
            )

        # 4. Finalize job status
        if job.failed_chunks > 0 or job.failed_papers > 0 or job.bm25_status == "FAILED":
            job.embedding_status = "PARTIAL" if job.embedded_chunks > 0 else "FAILED"
            job.status = JobStatus.COMPLETED # Not strictly failed if it did partial work, but flags indicate issue
        else:
            job.embedding_status = "COMPLETE"
            job.status = JobStatus.COMPLETED

    async def _update_paper_status(self, paper_id: str, status: EmbeddingStatus) -> None:
        """Update the embedding status on the paper record."""
        await self._paper_repo.update_embedding_status(paper_id, status.value)
