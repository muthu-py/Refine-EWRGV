"""
app/ingestion/representation/worker.py
--------------------------------------
Background coroutine that drives text representation (Phase 2.9) for all eligible
papers linked to a given research run.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.core.exceptions import StorageError
from app.core.job_store import JobStatus, RepresentationJob
from app.core.logging import get_logger
from app.ingestion.representation.service import RepresentationService

logger = get_logger(__name__)


async def run_representation(
    job: RepresentationJob,
    job_store: Any,  # JobStore
    representation_service: RepresentationService,
) -> None:
    """
    Background worker that orchestrates text representation for a run.
    """
    try:
        job.status = JobStatus.RUNNING
        await job_store.update_representation_job(job)
        logger.info("Starting representation worker for job %s (run %s)", job.job_id, job.run_id)

        await representation_service.run_representation_for_run(job.run_id, job)

    except Exception as exc:
        logger.error("Representation worker failed for job %s: %s", job.job_id, exc, exc_info=True)
        job.status = JobStatus.FAILED
        job.error = str(exc)

    finally:
        # If it wasn't already marked completed or failed by the service
        if job.status not in (JobStatus.COMPLETED, JobStatus.FAILED):
            job.status = JobStatus.COMPLETED
            
        await job_store.update_representation_job(job)
        logger.info(
            "Representation worker finished for job %s. Status: %s. "
            "Embedded: %d/%d chunks.",
            job.job_id,
            job.status.value,
            job.embedded_chunks,
            job.total_chunks,
        )
