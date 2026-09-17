"""
app/ingestion/chunking/worker.py
----------------------------------
Background coroutine that drives chunking (Phase 2.7) for all eligible
papers linked to a given research run.

Called via asyncio.create_task() from the POST /research/chunks route handler.

Eligibility criteria (enforced by repo.get_papers_for_chunking):
  - full_text_status = 'DOWNLOADED'
  - parsing_status   = 'PARSED'
  - section_status   = 'EXTRACTED'
  - chunk_status    != 'CHUNKED'

Sequence for each eligible paper:
  1. Download sections.json from Supabase Storage.
  2. Deserialise SectionExtractionResult.
  3. Run SectionChunker.chunk().
  4. Validate chunks (non-empty, within size).
  5. Serialise ChunkingResult to JSON.
  6. Upload chunks.json to Storage at papers/{paper_id}/chunks.json.
  7. Update chunk_status = 'CHUNKED' in DB.

Failure on one paper does not stop processing of subsequent papers.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from app.core.config import settings
from app.core.exceptions import StorageError
from app.core.job_store import ChunkingJob, JobStatus
from app.core.logging import get_logger
from app.domain.models.parsed_section import SectionExtractionResult
from app.ingestion.chunking.chunker import ChunkConfig, SectionChunker
from app.storage.artifact_paths import (
    ARTIFACT_BUCKET,
    get_chunks_path,
    get_sections_path,
)
from app.storage.repositories.paper_repository import PaperRepository
from app.storage.supabase_storage import SupabaseStorageClient

logger = get_logger(__name__)

_CHUNKS_JSON_FILENAME = "chunks.json"


async def run_chunking(
    job: ChunkingJob,
    job_store,                # JobStore — not typed to avoid circular import
    repo: PaperRepository,
) -> None:
    """
    Drive Phase 2.7 chunking for all eligible EXTRACTED papers.

    Idempotent: CHUNKED papers are excluded by get_papers_for_chunking().
    """
    job.status = JobStatus.RUNNING
    job.started_at = datetime.now(timezone.utc)
    await job_store.update_chunking_job(job)

    storage_client = SupabaseStorageClient()
    bucket = settings.SUPABASE_STORAGE_BUCKET

    # Build chunker config from application settings
    config = ChunkConfig(
        target_chunk_size=settings.CHUNK_TARGET_SIZE,
        max_chunk_size=settings.CHUNK_MAX_SIZE,
        overlap_size=settings.CHUNK_OVERLAP,
        min_chunk_size=settings.CHUNK_MIN_SIZE,
    )
    chunker = SectionChunker(config=config)

    try:
        papers = await repo.get_papers_for_chunking(job.run_id)
        job.total_papers = len(papers)
        await job_store.update_chunking_job(job)

        logger.info(
            "Chunking worker starting",
            extra={"job_id": job.job_id, "run_id": job.run_id, "total": job.total_papers},
        )

        for paper in papers:
            paper_id = paper["paper_id"]
            sections_path = paper.get("sections_storage_path") or ""

            if not sections_path:
                logger.warning(
                    "Chunking worker: skipping paper with no sections_storage_path",
                    extra={"paper_id": paper_id},
                )
                await repo.update_chunk_metadata(paper_id=paper_id, status="CHUNKING_FAILED")
                job.failed += 1
                await job_store.update_chunking_job(job)
                continue

            try:
                # --- Step 1: Download sections.json ---
                # Use artifact_paths for the canonical path; normalise legacy paths.
                normalised = _normalise_path(
                    sections_path or get_sections_path(paper_id), bucket
                )
                sections_bytes = await storage_client.download(bucket=bucket, path=normalised)

                # --- Step 2: Deserialise SectionExtractionResult ---
                sections_data = json.loads(sections_bytes.decode("utf-8"))
                extraction_result = SectionExtractionResult.model_validate(sections_data)

                # --- Step 3: Chunk ---
                result = chunker.chunk(
                    extraction_result=extraction_result,
                    source_sections_path=sections_path,
                )

                # --- Step 4: Validate chunks ---
                valid_chunks = [c for c in result.chunks if c.text.strip()]
                result.chunks = valid_chunks
                result.diagnostics.total_chunks = len(valid_chunks)

                # --- Step 5: Serialise to JSON bytes ---
                chunks_json_bytes = result.model_dump_json(indent=2).encode("utf-8")

                # --- Step 6: Upload chunks.json ---
                # Use artifact_paths for the canonical, centralized path.
                chunks_path = get_chunks_path(paper_id)
                await storage_client.upload(
                    bucket=bucket,
                    path=chunks_path,
                    content=chunks_json_bytes,
                    content_type="application/json",
                    upsert=True,
                )

                # --- Step 7: Update DB ---
                await repo.update_chunk_metadata(
                    paper_id=paper_id,
                    status="CHUNKED",
                    chunks_storage_path=chunks_path,
                    chunks_count=len(result.chunks),
                )

                job.chunked += 1
                logger.info(
                    "Chunking worker: paper chunked successfully",
                    extra={
                        "paper_id": paper_id,
                        "chunks": len(result.chunks),
                        "chunks_path": chunks_path,
                    },
                )

            except StorageError as exc:
                logger.warning(
                    "Chunking worker: storage error",
                    extra={"paper_id": paper_id, "error": str(exc)},
                )
                await _safe_fail(repo, paper_id)
                job.failed += 1

            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Chunking worker: unexpected error",
                    extra={"paper_id": paper_id, "error": str(exc)},
                )
                await _safe_fail(repo, paper_id)
                job.failed += 1

            await job_store.update_chunking_job(job)

        job.status = JobStatus.COMPLETED

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Chunking worker: fatal error",
            extra={"job_id": job.job_id, "error": str(exc)},
        )
        job.status = JobStatus.FAILED
        job.error = str(exc)

    finally:
        job.completed_at = datetime.now(timezone.utc)
        await job_store.update_chunking_job(job)
        await storage_client.close()

    logger.info(
        "Chunking worker complete",
        extra={"job_id": job.job_id, "chunked": job.chunked, "failed": job.failed},
    )


def _normalise_path(path: str, bucket: str) -> str:
    double = f"{bucket}/{bucket}/"
    if path.startswith(double):
        return path[len(f"{bucket}/"):]
    return path


async def _safe_fail(repo: PaperRepository, paper_id: str) -> None:
    try:
        await repo.update_chunk_metadata(paper_id=paper_id, status="CHUNKING_FAILED")
    except Exception:  # noqa: BLE001
        pass
