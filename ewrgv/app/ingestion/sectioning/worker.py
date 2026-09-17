"""
app/ingestion/sectioning/worker.py
-------------------------------------
Background coroutine that drives section extraction (Phase 2.6) for all
eligible papers linked to a given research run.

Called via asyncio.create_task() from the POST /research/sections route handler.

Eligibility criteria (enforced by repo.get_papers_for_sectioning):
  - full_text_status = 'DOWNLOADED'
  - parsing_status   = 'PARSED'
  - section_status  != 'EXTRACTED'

Sequence for each eligible paper:
  1. Download parsed.json from Supabase Storage.
  2. Deserialise ParsedDocument.
  3. Run SectionExtractor.extract().
  4. Serialise SectionExtractionResult to JSON.
  5. Upload sections.json to Storage at papers/{paper_id}/sections.json.
  6. Update section_status = 'EXTRACTED' in DB.

Failure on one paper does not stop processing of subsequent papers.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from app.core.exceptions import IngestionError, StorageError
from app.core.job_store import JobStatus, SectioningJob
from app.core.logging import get_logger
from app.domain.models.parsed_document import ParsedDocument
from app.ingestion.sectioning.extractor import SectionExtractor
from app.storage.artifact_paths import (
    ARTIFACT_BUCKET,
    get_parsed_path,
    get_sections_path,
)
from app.storage.repositories.paper_repository import PaperRepository
from app.storage.supabase_storage import SupabaseStorageClient
from app.core.config import settings

logger = get_logger(__name__)

_SECTIONS_JSON_FILENAME = "sections.json"


async def run_sectioning(
    job: SectioningJob,
    job_store,               # JobStore — not typed to avoid circular import
    repo: PaperRepository,
) -> None:
    """
    Drive Phase 2.6 section extraction for all eligible PARSED papers.

    Progress is written to job_store incrementally so callers can poll
    GET /research/sections/status/{job_id} for live updates.

    Idempotent: EXTRACTED papers are excluded by get_papers_for_sectioning().
    """
    job.status = JobStatus.RUNNING
    job.started_at = datetime.now(timezone.utc)
    await job_store.update_sectioning_job(job)

    storage_client = SupabaseStorageClient()
    bucket = settings.SUPABASE_STORAGE_BUCKET
    extractor = SectionExtractor()

    try:
        papers = await repo.get_papers_for_sectioning(job.run_id)
        job.total_papers = len(papers)
        await job_store.update_sectioning_job(job)

        logger.info(
            "Sectioning worker starting",
            extra={"job_id": job.job_id, "run_id": job.run_id, "total": job.total_papers},
        )

        for paper in papers:
            paper_id = paper["paper_id"]
            parsed_storage_path = paper.get("parsed_storage_path") or ""

            if not parsed_storage_path:
                logger.warning(
                    "Sectioning worker: skipping paper with no parsed_storage_path",
                    extra={"paper_id": paper_id},
                )
                await repo.update_section_metadata(
                    paper_id=paper_id,
                    status="EXTRACTION_FAILED",
                )
                job.failed += 1
                await job_store.update_sectioning_job(job)
                continue

            try:
                # --- Step 1: Download parsed.json ---
                # Use artifact_paths for the canonical path.
                # Also normalise any legacy double-prefix paths from old DB rows.
                normalised_path = _normalise_path(
                    parsed_storage_path or get_parsed_path(paper_id), bucket
                )
                parsed_bytes = await storage_client.download(bucket=bucket, path=normalised_path)

                # --- Step 2: Deserialise ParsedDocument ---
                parsed_data = json.loads(parsed_bytes.decode("utf-8"))
                parsed_doc = ParsedDocument.model_validate(parsed_data)

                # --- Step 3: Extract sections ---
                result = extractor.extract(
                    doc=parsed_doc,
                    source_parsed_path=parsed_storage_path,
                )

                # --- Step 4: Serialise to JSON bytes ---
                sections_json_bytes = result.model_dump_json(indent=2).encode("utf-8")

                # --- Step 5: Upload sections.json ---
                # Use artifact_paths to get the canonical, centralized path.
                sections_path = get_sections_path(paper_id)
                await storage_client.upload(
                    bucket=bucket,
                    path=sections_path,
                    content=sections_json_bytes,
                    content_type="application/json",
                    upsert=True,
                )

                # --- Step 6: Update DB ---
                await repo.update_section_metadata(
                    paper_id=paper_id,
                    status="EXTRACTED",
                    sections_storage_path=sections_path,
                    sections_count=len(result.sections),
                )

                job.extracted += 1
                logger.info(
                    "Sectioning worker: paper extracted successfully",
                    extra={
                        "paper_id": paper_id,
                        "sections": len(result.sections),
                        "sections_path": sections_path,
                        "fallback": result.diagnostics.fallback_used,
                    },
                )

            except (IngestionError, StorageError) as exc:
                logger.warning(
                    "Sectioning worker: error processing paper",
                    extra={"paper_id": paper_id, "error": str(exc)},
                )
                await _safe_fail(repo, paper_id)
                job.failed += 1

            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Sectioning worker: unexpected error processing paper",
                    extra={"paper_id": paper_id, "error": str(exc)},
                )
                await _safe_fail(repo, paper_id)
                job.failed += 1

            await job_store.update_sectioning_job(job)

        job.status = JobStatus.COMPLETED

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Sectioning worker: fatal error",
            extra={"job_id": job.job_id, "error": str(exc)},
        )
        job.status = JobStatus.FAILED
        job.error = str(exc)

    finally:
        job.completed_at = datetime.now(timezone.utc)
        await job_store.update_sectioning_job(job)
        await storage_client.close()

    logger.info(
        "Sectioning worker complete",
        extra={
            "job_id": job.job_id,
            "extracted": job.extracted,
            "failed": job.failed,
        },
    )


def _normalise_path(path: str, bucket: str) -> str:
    """Strip double bucket prefix if present (legacy Phase 2.4 bug workaround)."""
    double = f"{bucket}/{bucket}/"
    if path.startswith(double):
        return path[len(f"{bucket}/"):]
    return path


async def _safe_fail(repo: PaperRepository, paper_id: str) -> None:
    """Best-effort DB update marking section extraction as failed."""
    try:
        await repo.update_section_metadata(paper_id=paper_id, status="EXTRACTION_FAILED")
    except Exception:  # noqa: BLE001
        pass
