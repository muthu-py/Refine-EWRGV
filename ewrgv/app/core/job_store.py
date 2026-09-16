"""
app/core/job_store.py
---------------------
In-memory job store for tracking background acquisition jobs.

Design notes
------------
* Jobs are held in a plain dict protected by an asyncio.Lock.
* The store is mounted on app.state at startup so all routes share one instance.
* In-memory only — jobs are lost on server restart. Acceptable for Phase 2.4.
  A persistent `acquisition_jobs` table can replace this in a future phase.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class AcquisitionJob:
    """Tracks the state of one full-text acquisition background job."""
    job_id: str
    run_id: str
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_papers: int = 0
    acquired: int = 0
    failed: int = 0
    no_oa: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "run_id": self.run_id,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "total_papers": self.total_papers,
            "acquired": self.acquired,
            "failed": self.failed,
            "no_oa": self.no_oa,
            "error": self.error,
        }


class JobStore:
    """Thread-safe in-memory store for AcquisitionJob objects."""

    def __init__(self) -> None:
        self._jobs: dict[str, AcquisitionJob] = {}
        self._lock = asyncio.Lock()

    async def create_job(self, run_id: str) -> AcquisitionJob:
        job = AcquisitionJob(job_id=str(uuid4()), run_id=run_id)
        async with self._lock:
            self._jobs[job.job_id] = job
        return job

    async def get_job(self, job_id: str) -> Optional[AcquisitionJob]:
        async with self._lock:
            return self._jobs.get(job_id)

    async def update_job(self, job: AcquisitionJob) -> None:
        async with self._lock:
            self._jobs[job.job_id] = job
