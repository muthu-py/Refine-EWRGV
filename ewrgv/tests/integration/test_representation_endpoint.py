import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

from app.core.job_store import JobStatus, RepresentationJob
from app.main import app

client = TestClient(app)

# ---------------------------------------------------------------------------
# Test Data & Fixtures
# ---------------------------------------------------------------------------

DUMMY_RUN_ID = "run_rep_1"
DUMMY_JOB_ID = "job_rep_1"


@pytest.fixture
def mock_job_store(monkeypatch):
    """Mocks the app.state.job_store so we don't need real DB or background tasks."""
    class MockJobStore:
        async def create_representation_job(self, run_id: str):
            job = RepresentationJob(job_id=DUMMY_JOB_ID, run_id=run_id)
            self.job = job
            return job

        async def get_representation_job(self, job_id: str):
            if hasattr(self, "job") and self.job.job_id == job_id:
                return self.job
            return None

        async def update_representation_job(self, job):
            self.job = job

    store = MockJobStore()
    monkeypatch.setattr(app.state, "job_store", store, raising=False)
    return store


@pytest.fixture
def disable_background_tasks(monkeypatch):
    """Prevents the background task from actually running and failing due to lack of DB."""
    import asyncio
    monkeypatch.setattr(asyncio, "create_task", lambda *args, **kwargs: None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_start_representation(mock_job_store, disable_background_tasks):
    # We mock the dependencies created in the route handler so it doesn't try to connect to the DB
    with pytest.MonkeyPatch.context() as m:
        m.setattr("app.storage.relational.database.get_db_pool", AsyncMock(return_value=None))
        m.setattr("app.storage.repositories.paper_repository.PaperRepository", lambda *args: None)
        m.setattr("app.storage.repositories.embedding_repository.EmbeddingRepository", lambda *args: None)
        m.setattr("app.storage.supabase_storage.SupabaseStorageClient", lambda *args: None)
        m.setattr("app.providers.embeddings.factory.create_embedding_provider", lambda *args: None)
        m.setattr("app.ingestion.representation.service.RepresentationService", lambda *args, **kwargs: None)
        
        response = client.post(
            "/api/v1/research/represent",
            json={"run_id": DUMMY_RUN_ID},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == DUMMY_JOB_ID
        assert data["run_id"] == DUMMY_RUN_ID
        assert data["status"] == JobStatus.PENDING.value
        assert "started in background" in data["message"]


def test_get_representation_status(mock_job_store):
    # Setup a job in the mock store
    job = RepresentationJob(
        job_id=DUMMY_JOB_ID,
        run_id=DUMMY_RUN_ID,
        status=JobStatus.RUNNING,
        total_papers=10,
        processed_papers=5,
        total_chunks=100,
        embedded_chunks=50,
        embedding_status="PARTIAL",
        bm25_status="NOT_INDEXED"
    )
    mock_job_store.job = job

    response = client.get(f"/api/v1/research/represent/status/{DUMMY_JOB_ID}")
    
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == DUMMY_JOB_ID
    assert data["run_id"] == DUMMY_RUN_ID
    assert data["status"] == JobStatus.RUNNING.value
    assert data["total_papers"] == 10
    assert data["processed_papers"] == 5
    assert data["total_chunks"] == 100
    assert data["embedded_chunks"] == 50
    assert data["embedding_status"] == "PARTIAL"
    assert data["bm25_status"] == "NOT_INDEXED"


def test_get_representation_status_not_found(mock_job_store):
    response = client.get("/api/v1/research/represent/status/nonexistent_job")
    
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"]
