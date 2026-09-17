"""
app/core/job_store.py
---------------------
In-memory job store for tracking all EWRGV background pipeline jobs.

Design notes
------------
* Jobs are held in plain dicts protected by an asyncio.Lock.
* The store is mounted on app.state at startup so all routes share one instance.
* In-memory only — jobs are lost on server restart. Acceptable for Phases 2.4–2.7.
  A persistent jobs table can replace this in a future phase.
* Phase 2.4: AcquisitionJob / create_job / get_job
* Phase 2.5: ParsingJob / create_parsing_job / get_parsing_job
* Phase 2.6: SectioningJob / create_sectioning_job / get_sectioning_job
* Phase 2.7: ChunkingJob / create_chunking_job / get_chunking_job
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


@dataclass
class ParsingJob:
    """Tracks the state of one document-parsing background job (Phase 2.5)."""
    job_id: str
    run_id: str
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_papers: int = 0
    parsed: int = 0
    failed: int = 0
    skipped: int = 0
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
            "parsed": self.parsed,
            "failed": self.failed,
            "skipped": self.skipped,
            "error": self.error,
        }


@dataclass
class SectioningJob:
    """Tracks the state of one section-extraction background job (Phase 2.6)."""
    job_id: str
    run_id: str
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_papers: int = 0
    extracted: int = 0
    failed: int = 0
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
            "extracted": self.extracted,
            "failed": self.failed,
            "error": self.error,
        }


@dataclass
class ChunkingJob:
    """Tracks the state of one chunking background job (Phase 2.7)."""
    job_id: str
    run_id: str
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_papers: int = 0
    chunked: int = 0
    failed: int = 0
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
            "chunked": self.chunked,
            "failed": self.failed,
            "error": self.error,
        }


@dataclass
class RepresentationJob:
    """Tracks the state of one representation background job (Phase 2.9)."""
    job_id: str
    run_id: str
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_papers: int = 0
    processed_papers: int = 0
    failed_papers: int = 0
    total_chunks: int = 0
    embedded_chunks: int = 0
    failed_chunks: int = 0
    embedding_status: str = "NOT_EMBEDDED"
    bm25_status: str = "NOT_INDEXED"
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
            "processed_papers": self.processed_papers,
            "failed_papers": self.failed_papers,
            "total_chunks": self.total_chunks,
            "embedded_chunks": self.embedded_chunks,
            "failed_chunks": self.failed_chunks,
            "embedding_status": self.embedding_status,
            "bm25_status": self.bm25_status,
            "error": self.error,
        }



class JobStore:
    """Thread-safe in-memory store for all EWRGV background pipeline jobs."""

    def __init__(self) -> None:
        self._jobs: dict[str, AcquisitionJob] = {}
        self._parsing_jobs: dict[str, ParsingJob] = {}
        self._sectioning_jobs: dict[str, SectioningJob] = {}
        self._chunking_jobs: dict[str, ChunkingJob] = {}
        self._representation_jobs: dict[str, RepresentationJob] = {}
        self._lock = asyncio.Lock()

    # ---- Phase 2.4: Acquisition jobs (unchanged) ---- #

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

    # ---- Phase 2.5: Parsing jobs ---- #

    async def create_parsing_job(self, run_id: str) -> ParsingJob:
        """Create and register a new ParsingJob for the given run_id."""
        job = ParsingJob(job_id=str(uuid4()), run_id=run_id)
        async with self._lock:
            self._parsing_jobs[job.job_id] = job
        return job

    async def get_parsing_job(self, job_id: str) -> Optional[ParsingJob]:
        """Retrieve a ParsingJob by its job_id."""
        async with self._lock:
            return self._parsing_jobs.get(job_id)

    async def update_parsing_job(self, job: ParsingJob) -> None:
        """Persist an updated ParsingJob state."""
        async with self._lock:
            self._parsing_jobs[job.job_id] = job

    # ---- Phase 2.6: Sectioning jobs ---- #

    async def create_sectioning_job(self, run_id: str) -> SectioningJob:
        """Create and register a new SectioningJob for the given run_id."""
        job = SectioningJob(job_id=str(uuid4()), run_id=run_id)
        async with self._lock:
            self._sectioning_jobs[job.job_id] = job
        return job

    async def get_sectioning_job(self, job_id: str) -> Optional[SectioningJob]:
        """Retrieve a SectioningJob by its job_id."""
        async with self._lock:
            return self._sectioning_jobs.get(job_id)

    async def update_sectioning_job(self, job: SectioningJob) -> None:
        """Persist an updated SectioningJob state."""
        async with self._lock:
            self._sectioning_jobs[job.job_id] = job

    # ---- Phase 2.7: Chunking jobs ---- #

    async def create_chunking_job(self, run_id: str) -> ChunkingJob:
        """Create and register a new ChunkingJob for the given run_id."""
        job = ChunkingJob(job_id=str(uuid4()), run_id=run_id)
        async with self._lock:
            self._chunking_jobs[job.job_id] = job
        return job

    async def get_chunking_job(self, job_id: str) -> Optional[ChunkingJob]:
        """Retrieve a ChunkingJob by its job_id."""
        async with self._lock:
            return self._chunking_jobs.get(job_id)

    async def update_chunking_job(self, job: ChunkingJob) -> None:
        """Persist an updated ChunkingJob state."""
        async with self._lock:
            self._chunking_jobs[job.job_id] = job

    # ---- Phase 2.9: Representation jobs ---- #

    async def create_representation_job(self, run_id: str) -> RepresentationJob:
        """Create and register a new RepresentationJob for the given run_id."""
        job = RepresentationJob(job_id=str(uuid4()), run_id=run_id)
        async with self._lock:
            self._representation_jobs[job.job_id] = job
        return job

    async def get_representation_job(self, job_id: str) -> Optional[RepresentationJob]:
        """Retrieve a RepresentationJob by its job_id."""
        async with self._lock:
            return self._representation_jobs.get(job_id)

    async def update_representation_job(self, job: RepresentationJob) -> None:
        """Persist an updated RepresentationJob state."""
        async with self._lock:
            self._representation_jobs[job.job_id] = job

