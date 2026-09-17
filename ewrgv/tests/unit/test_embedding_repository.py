import pytest
from unittest.mock import AsyncMock, MagicMock

from app.storage.repositories.embedding_repository import (
    EmbeddingRecord,
    EmbeddingRepository,
)

# ---------------------------------------------------------------------------
# Test Data
# ---------------------------------------------------------------------------

DUMMY_MODEL = "test-model"
DUMMY_DIMENSION = 3


def get_mock_pool():
    pool = MagicMock()
    conn = AsyncMock()
    # Mock the context manager behavior for acquire()
    pool.acquire.return_value.__aenter__.return_value = conn
    return pool, conn


@pytest.fixture
def repo_and_conn():
    pool, conn = get_mock_pool()
    repo = EmbeddingRepository(pool)
    return repo, conn


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_save_embeddings(repo_and_conn):
    repo, conn = repo_and_conn
    
    records = [
        EmbeddingRecord("chunk1", "paper1", [0.1, 0.2, 0.3], DUMMY_MODEL, DUMMY_DIMENSION),
        EmbeddingRecord("chunk2", "paper1", [0.4, 0.5, 0.6], DUMMY_MODEL, DUMMY_DIMENSION),
    ]
    
    await repo.save_embeddings(records)
    
    conn.executemany.assert_called_once()
    args, kwargs = conn.executemany.call_args
    query = args[0]
    values = args[1]
    
    assert "INSERT INTO chunk_embeddings" in query
    assert "ON CONFLICT (chunk_id, embedding_model) DO UPDATE" in query
    
    assert len(values) == 2
    # Verify the structure of the executemany arguments (chunk_id, paper_id, json_embedding, model, dim)
    assert values[0][0] == "chunk1"
    assert values[0][1] == "paper1"
    assert values[0][3] == DUMMY_MODEL
    assert values[0][4] == DUMMY_DIMENSION
    assert values[1][0] == "chunk2"


@pytest.mark.asyncio
async def test_save_embeddings_empty(repo_and_conn):
    repo, conn = repo_and_conn
    await repo.save_embeddings([])
    conn.executemany.assert_not_called()


@pytest.mark.asyncio
async def test_get_embedded_chunk_ids(repo_and_conn):
    repo, conn = repo_and_conn
    
    conn.fetch.return_value = [
        {"chunk_id": "chunk1"},
        {"chunk_id": "chunk2"},
    ]
    
    result = await repo.get_embedded_chunk_ids("paper1", DUMMY_MODEL)
    
    assert result == {"chunk1", "chunk2"}
    conn.fetch.assert_called_once()
    args, _ = conn.fetch.call_args
    assert args[1] == "paper1"
    assert args[2] == DUMMY_MODEL


@pytest.mark.asyncio
async def test_get_embedding_counts(repo_and_conn):
    repo, conn = repo_and_conn
    
    conn.fetchval.return_value = 5
    
    result = await repo.get_embedding_counts("paper1", DUMMY_MODEL)
    
    assert result == 5
    conn.fetchval.assert_called_once()
    args, _ = conn.fetchval.call_args
    assert args[1] == "paper1"
    assert args[2] == DUMMY_MODEL


@pytest.mark.asyncio
async def test_delete_embeddings(repo_and_conn):
    repo, conn = repo_and_conn
    
    chunk_ids = ["chunk1", "chunk2"]
    await repo.delete_embeddings(chunk_ids, DUMMY_MODEL)
    
    conn.execute.assert_called_once()
    args, _ = conn.execute.call_args
    assert "DELETE FROM chunk_embeddings" in args[0]
    assert args[1] == chunk_ids
    assert args[2] == DUMMY_MODEL


@pytest.mark.asyncio
async def test_delete_embeddings_empty(repo_and_conn):
    repo, conn = repo_and_conn
    await repo.delete_embeddings([], DUMMY_MODEL)
    conn.execute.assert_not_called()
