"""
app/ingestion/parsers/worker.py
---------------------------------
Background coroutine that drives document parsing (Phase 2.5) for all
eligible papers linked to a given research run.

Called via asyncio.create_task() from the POST /research/parse route handler.

Eligibility criteria (enforced by repo.get_papers_for_parsing):
  - full_text_status = 'DOWNLOADED'   (PDF has been successfully acquired)
  - parsing_status   != 'PARSED'      (not already parsed; allows retry of PARSE_FAILED)

The sequence for each eligible paper:
  1. Download PDF from Supabase Storage via PDFParser.
  2. Extract page-level text.
  3. Detect low-text / invalid documents.
  4. Serialise ParsedDocument to JSON (UTF-8).
  5. Upload parsed.json to Storage at papers/{paper_id}/parsed.json.
  6. Update parsing_status = 'PARSED' in DB.

If any step fails for a single paper:
  - Log the error (without credentials).
  - Mark parsing_status = 'PARSE_FAILED'.
  - Increment job.failed.
  - Continue to the next paper.

A systemic failure (e.g. DB unavailable) sets job.status = FAILED.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from app.core.exceptions import IngestionError, StorageError
from app.core.job_store import JobStatus, ParsingJob
from app.core.logging import get_logger
from app.ingestion.parsers.pdf_parser import PDFParser, _MIN_MEANINGFUL_CHARS
from app.storage.artifact_paths import ARTIFACT_BUCKET, get_parsed_path
from app.storage.repositories.paper_repository import PaperRepository
from app.storage.supabase_storage import SupabaseStorageClient
from app.core.config import settings

logger = get_logger(__name__)

# Kept for backward compat reference only; actual path built via artifact_paths.
_PARSED_JSON_FILENAME = "parsed.json"


async def run_parsing(
    job: ParsingJob,
    job_store,               # JobStore — not typed to avoid circular import
    repo: PaperRepository,
) -> None:
    """
    Drive Phase 2.5 parsing for all eligible DOWNLOADED papers in a research run.

    Progress is written to `job_store` incrementally so callers can poll
    GET /research/parse/status/{job_id} and see live updates.

    Idempotent: papers already PARSED are excluded by get_papers_for_parsing().
    """
    job.status = JobStatus.RUNNING
    job.started_at = datetime.now(timezone.utc)
    await job_store.update_parsing_job(job)

    storage_client = SupabaseStorageClient()
    bucket = settings.SUPABASE_STORAGE_BUCKET
    parser = PDFParser(storage_client=storage_client, bucket=bucket)

    try:
        papers = await repo.get_papers_for_parsing(job.run_id)
        job.total_papers = len(papers)
        await job_store.update_parsing_job(job)

        logger.info(
            "Parsing worker starting",
            extra={"job_id": job.job_id, "run_id": job.run_id, "total": job.total_papers},
        )

        for paper in papers:
            paper_id = paper["paper_id"]
            storage_path = paper.get("storage_path") or ""

            if not storage_path:
                # No storage path recorded — cannot parse this paper
                logger.warning(
                    "Parsing worker: skipping paper with no storage_path",
                    extra={"paper_id": paper_id},
                )
                await repo.update_parsing_metadata(
                    paper_id=paper_id,
                    status="PARSE_FAILED",
                )
                job.failed += 1
                await job_store.update_parsing_job(job)
                continue

            try:
                # --- Step 1 & 2: Download + parse PDF ---
                parsed_doc = await parser.parse(paper_id=paper_id, storage_path=storage_path)

                # --- Step 3: Detect low-text / invalid documents ---
                if parsed_doc.total_character_count < _MIN_MEANINGFUL_CHARS:
                    logger.warning(
                        "Parsing worker: low-text document detected",
                        extra={
                            "paper_id": paper_id,
                            "total_chars": parsed_doc.total_character_count,
                            "page_count": parsed_doc.page_count,
                        },
                    )
                    await repo.update_parsing_metadata(
                        paper_id=paper_id,
                        status="INVALID_DOCUMENT",
                        page_count=parsed_doc.page_count,
                    )
                    job.failed += 1
                    await job_store.update_parsing_job(job)
                    continue

                # --- Step 4: Serialise to JSON bytes (UTF-8) ---
                parsed_json_bytes = parsed_doc.model_dump_json(indent=2).encode("utf-8")

                # --- Step 5: Upload parsed.json to Storage ---
                # Use artifact_paths to get the canonical, centralized path.
                parsed_path = get_parsed_path(paper_id)
                await storage_client.upload(
                    bucket=bucket,
                    path=parsed_path,
                    content=parsed_json_bytes,
                    content_type="application/json",
                    upsert=True,
                )

                # --- Step 6: Update DB only after successful upload ---
                await repo.update_parsing_metadata(
                    paper_id=paper_id,
                    status="PARSED",
                    parsed_storage_path=parsed_path,
                    page_count=parsed_doc.page_count,
                )

                job.parsed += 1
                logger.info(
                    "Parsing worker: paper parsed successfully",
                    extra={
                        "paper_id": paper_id,
                        "page_count": parsed_doc.page_count,
                        "total_chars": parsed_doc.total_character_count,
                        "parsed_path": parsed_path,
                    },
                )

            except (IngestionError, StorageError) as exc:
                logger.warning(
                    "Parsing worker: error processing paper",
                    extra={"paper_id": paper_id, "error": str(exc)},
                )
                await repo.update_parsing_metadata(
                    paper_id=paper_id,
                    status="PARSE_FAILED",
                )
                job.failed += 1

            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Parsing worker: unexpected error processing paper",
                    extra={"paper_id": paper_id, "error": str(exc)},
                )
                try:
                    await repo.update_parsing_metadata(
                        paper_id=paper_id,
                        status="PARSE_FAILED",
                    )
                except Exception:  # noqa: BLE001
                    pass  # Don't let a DB update failure mask the original error
                job.failed += 1

            await job_store.update_parsing_job(job)

        job.status = JobStatus.COMPLETED

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Parsing worker: fatal error",
            extra={"job_id": job.job_id, "error": str(exc)},
        )
        job.status = JobStatus.FAILED
        job.error = str(exc)

    finally:
        job.completed_at = datetime.now(timezone.utc)
        await job_store.update_parsing_job(job)
        await storage_client.close()

    logger.info(
        "Parsing worker complete",
        extra={
            "job_id": job.job_id,
            "parsed": job.parsed,
            "failed": job.failed,
            "skipped": job.skipped,
        },
    )
