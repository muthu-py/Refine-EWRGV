"""
app/ingestion/acquisition/worker.py
------------------------------------
Background coroutine that drives OA discovery + full-text acquisition
for all papers linked to a given research run.

Called via asyncio.create_task() from the /acquire route handler.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from app.core.job_store import AcquisitionJob, JobStatus
from app.core.logging import get_logger
from app.ingestion.acquisition.discovery import OADiscoveryService
from app.ingestion.acquisition.fulltext import FullTextAcquisitionService
from app.storage.repositories.paper_repository import PaperRepository

logger = get_logger(__name__)


async def run_acquisition(
    job: AcquisitionJob,
    job_store,               # JobStore — not typed to avoid circular import
    repo: PaperRepository,
) -> None:
    """
    Drive Phase 2.4 acquisition for all NOT_CHECKED papers in a research run.

    Progress is written to `job_store` incrementally so callers can poll
    GET /{job_id} and see live updates.
    Idempotent: papers already DOWNLOADED are skipped automatically because
    `get_papers_for_run()` only returns NOT_CHECKED rows.
    """
    # Mark job as running
    job.status = JobStatus.RUNNING
    job.started_at = datetime.now(timezone.utc)
    await job_store.update_job(job)

    discovery = OADiscoveryService()
    acquisition = FullTextAcquisitionService()

    try:
        papers = await repo.get_papers_for_run(job.run_id)
        job.total_papers = len(papers)
        await job_store.update_job(job)

        logger.info(
            "Acquisition worker starting",
            extra={"job_id": job.job_id, "run_id": job.run_id, "total": job.total_papers},
        )

        for paper in papers:
            try:
                oa_url, source_type = await discovery.discover(paper)

                if not oa_url:
                    await repo.update_fulltext_metadata(
                        paper_id=paper["paper_id"],
                        status="NO_OA_FOUND",
                    )
                    job.no_oa += 1
                    await job_store.update_job(job)
                    continue

                result = await acquisition.acquire(paper["paper_id"], oa_url)
                await repo.update_fulltext_metadata(
                    paper_id=paper["paper_id"],
                    status=result["status"],
                    full_text_url=oa_url,
                    storage_path=result["storage_path"],
                    source_type=source_type,
                    content_type=result["content_type"],
                    file_size=result["file_size"],
                )

                if result["status"] == "DOWNLOADED":
                    job.acquired += 1
                else:
                    job.failed += 1

            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Acquisition worker: error processing paper",
                    extra={"paper_id": paper.get("paper_id"), "error": str(exc)},
                )
                job.failed += 1

            await job_store.update_job(job)

        job.status = JobStatus.COMPLETED

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Acquisition worker: fatal error",
            extra={"job_id": job.job_id, "error": str(exc)},
        )
        job.status = JobStatus.FAILED
        job.error = str(exc)

    finally:
        job.completed_at = datetime.now(timezone.utc)
        await job_store.update_job(job)
        await discovery.close()
        await acquisition.close()

    logger.info(
        "Acquisition worker complete",
        extra={
            "job_id": job.job_id,
            "acquired": job.acquired,
            "failed": job.failed,
            "no_oa": job.no_oa,
        },
    )
