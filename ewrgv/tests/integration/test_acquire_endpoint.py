"""
tests/integration/test_acquire_endpoint.py
-------------------------------------------
Tests for POST /research/acquire and GET /research/{job_id}.
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from app.main import create_app
from app.core.job_store import JobStore, JobStatus, AcquisitionJob


@pytest.fixture
def app_with_job_store():
    """Create a test app with a real JobStore mounted."""
    app = create_app()
    # The lifespan mounts job_store on startup; for TestClient we set it manually
    app.state.job_store = JobStore()
    return app


@pytest.fixture
def client(app_with_job_store):
    return TestClient(app_with_job_store, raise_server_exceptions=True)


# ------------------------------------------------------------------ #
# JobStore unit tests
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_job_store_create_and_get():
    store = JobStore()
    job = await store.create_job("run-123")

    assert job.run_id == "run-123"
    assert job.status == JobStatus.PENDING
    assert job.job_id

    retrieved = await store.get_job(job.job_id)
    assert retrieved is not None
    assert retrieved.job_id == job.job_id


@pytest.mark.asyncio
async def test_job_store_get_missing():
    store = JobStore()
    result = await store.get_job("non-existent-id")
    assert result is None


@pytest.mark.asyncio
async def test_job_store_update():
    store = JobStore()
    job = await store.create_job("run-456")
    job.status = JobStatus.RUNNING
    job.acquired = 5
    await store.update_job(job)

    updated = await store.get_job(job.job_id)
    assert updated.status == JobStatus.RUNNING
    assert updated.acquired == 5


# ------------------------------------------------------------------ #
# /acquire endpoint tests (mocking DB)
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_start_acquisition_unknown_run_id(app_with_job_store):
    """A run_id that does not exist in the DB should return 404."""
    mock_pool = MagicMock()
    mock_conn = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    mock_conn.fetchval = AsyncMock(return_value=None)  # run_exists -> False

    async def fake_get_pool():
        return mock_pool

    with patch("app.storage.relational.database.get_db_pool", new=fake_get_pool):
        client = TestClient(app_with_job_store, raise_server_exceptions=False)
        resp = client.post("/api/v1/research/acquire", json={"run_id": "fake-run-id"})
        # With mocked pool returning None from fetchval, run_exists returns False → 404
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_acquire_response_schema_and_job_status():
    """Test that /acquire returns PENDING job and GET /{job_id} returns it."""
    from app.core.job_store import JobStore, AcquisitionJob, JobStatus

    store = JobStore()
    job = await store.create_job("run-789")
    retrieved = await store.get_job(job.job_id)

    d = retrieved.to_dict()
    assert d["job_id"] == job.job_id
    assert d["run_id"] == "run-789"
    assert d["status"] == "PENDING"
    assert d["acquired"] == 0
    assert d["failed"] == 0
    assert d["no_oa"] == 0


# ------------------------------------------------------------------ #
# GET /{job_id} tests
# ------------------------------------------------------------------ #

def test_get_job_status_not_found(client):
    """Polling for a non-existent job should return stage=not_found."""
    resp = client.get("/api/v1/research/nonexistent-job-id")
    assert resp.status_code == 200
    data = resp.json()
    assert data["stage"] == "not_found"


@pytest.mark.asyncio
async def test_get_job_status_real_job(app_with_job_store):
    """Manually insert a job into the store and poll it."""
    store = app_with_job_store.state.job_store
    job = await store.create_job("run-real")
    job.status = JobStatus.COMPLETED
    job.acquired = 3
    job.total_papers = 5
    await store.update_job(job)

    client = TestClient(app_with_job_store)
    resp = client.get(f"/api/v1/research/{job.job_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "COMPLETED"
    assert data["stage"] == "acquisition"
    assert data["result_summary"]["acquired"] == 3
    assert data["result_summary"]["total_papers"] == 5


# ------------------------------------------------------------------ #
# Acquisition worker unit tests
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_worker_marks_job_completed():
    """Worker should mark job COMPLETED after processing all papers."""
    from app.ingestion.acquisition.worker import run_acquisition

    store = JobStore()
    job = await store.create_job("run-worker")

    mock_repo = MagicMock()
    mock_repo.get_papers_for_run = AsyncMock(return_value=[
        {"paper_id": "p1", "doi": None, "full_text_url": None, "source_type": None, "full_text_status": "NOT_CHECKED"}
    ])
    mock_repo.update_fulltext_metadata = AsyncMock()

    with patch("app.ingestion.acquisition.discovery.OADiscoveryService.discover", new=AsyncMock(return_value=(None, None))):
        with patch("app.ingestion.acquisition.discovery.OADiscoveryService.close", new=AsyncMock()):
            with patch("app.ingestion.acquisition.fulltext.FullTextAcquisitionService.close", new=AsyncMock()):
                await run_acquisition(job=job, job_store=store, repo=mock_repo)

    final = await store.get_job(job.job_id)
    assert final.status == JobStatus.COMPLETED
    assert final.no_oa == 1
    assert final.acquired == 0


@pytest.mark.asyncio
async def test_worker_counts_acquired_paper():
    """Worker should increment acquired count when PDF downloads successfully."""
    from app.ingestion.acquisition.worker import run_acquisition

    store = JobStore()
    job = await store.create_job("run-worker-2")

    mock_repo = MagicMock()
    mock_repo.get_papers_for_run = AsyncMock(return_value=[
        {"paper_id": "p2", "doi": "10.1/test", "full_text_url": "https://pdf.url", "source_type": "openalex", "full_text_status": "NOT_CHECKED"}
    ])
    mock_repo.update_fulltext_metadata = AsyncMock()

    with patch("app.ingestion.acquisition.discovery.OADiscoveryService.discover", new=AsyncMock(return_value=("https://pdf.url", "openalex"))):
        with patch("app.ingestion.acquisition.fulltext.FullTextAcquisitionService.acquire", new=AsyncMock(return_value={"status": "DOWNLOADED", "storage_path": "papers/p2/fulltext.pdf", "content_type": "application/pdf", "file_size": 1024})):
            with patch("app.ingestion.acquisition.discovery.OADiscoveryService.close", new=AsyncMock()):
                with patch("app.ingestion.acquisition.fulltext.FullTextAcquisitionService.close", new=AsyncMock()):
                    await run_acquisition(job=job, job_store=store, repo=mock_repo)

    final = await store.get_job(job.job_id)
    assert final.status == JobStatus.COMPLETED
    assert final.acquired == 1
    assert final.no_oa == 0
