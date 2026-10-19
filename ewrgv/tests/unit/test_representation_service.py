import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.job_store import JobStatus, RepresentationJob
from app.domain.enums import EmbeddingStatus
from app.domain.models.paper_chunk import ChunkingDiagnostics, ChunkingResult, PaperChunk
from app.ingestion.representation.service import RepresentationService

# ---------------------------------------------------------------------------
# Test Data
# ---------------------------------------------------------------------------

DUMMY_RUN_ID = "run1"
DUMMY_PAPER_ID = "paper1"
DUMMY_MODEL = "test-model"


def create_dummy_chunk_result(paper_id: str, count: int) -> str:
    chunks = [
        PaperChunk(
            chunk_id=f"chunk{i}",
            paper_id=paper_id,
            section_id="sec1",
            section_title="Test",
            text=f"This is chunk {i}",
        )
        for i in range(count)
    ]
    result = ChunkingResult(
        paper_id=paper_id,
        source_sections_path="fake",
        chunks=chunks,
        diagnostics=ChunkingDiagnostics(),
    )
    return result.model_dump_json()


@pytest.fixture
def mocks():
    provider = AsyncMock()
    provider.get_model_name.return_value = DUMMY_MODEL
    provider.get_dimension.return_value = 3
    provider.embed_documents.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    embedding_repo = AsyncMock()
    embedding_repo.get_embedded_chunk_ids.return_value = set()
    embedding_repo.get_embedding_counts.return_value = 2

    paper_repo = AsyncMock()
    paper_repo.get_papers_for_representation.return_value = [
        {"paper_id": DUMMY_PAPER_ID, "chunks_storage_path": "path", "chunks_count": 2}
    ]

    storage_client = AsyncMock()
    storage_client.download.return_value = create_dummy_chunk_result(DUMMY_PAPER_ID, 2).encode("utf-8")

    return provider, embedding_repo, paper_repo, storage_client


@pytest.fixture
def service(mocks):
    provider, embedding_repo, paper_repo, storage_client = mocks
    return RepresentationService(
        embedding_provider=provider,
        embedding_repo=embedding_repo,
        paper_repo=paper_repo,
        storage_client=storage_client,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_representation_success(service, mocks):
    provider, embedding_repo, paper_repo, storage_client = mocks
    job = RepresentationJob(job_id="job1", run_id=DUMMY_RUN_ID)

    with patch("app.ingestion.representation.service.BM25Index.save") as mock_save:
        await service.run_representation_for_run(DUMMY_RUN_ID, job)

        # Provider called
        provider.embed_documents.assert_called_once()
        args, _ = provider.embed_documents.call_args
        assert len(args[0]) == 2
        
        # Embeddings saved
        embedding_repo.save_embeddings.assert_called_once()
        args, _ = embedding_repo.save_embeddings.call_args
        assert len(args[0]) == 2
        
        # Status updated
        paper_repo.update_embedding_status.assert_called_once_with(DUMMY_PAPER_ID, EmbeddingStatus.EMBEDDED)
        
        # BM25 saved
        mock_save.assert_called_once()

        assert job.status == JobStatus.COMPLETED
        assert job.embedding_status == "COMPLETE"
        assert job.bm25_status == "INDEXED"
        assert job.total_chunks == 2
        assert job.embedded_chunks == 2


@pytest.mark.asyncio
async def test_run_representation_incremental(service, mocks):
    provider, embedding_repo, paper_repo, storage_client = mocks
    
    # Simulate that chunk0 is already embedded, only chunk1 needs it
    embedding_repo.get_embedded_chunk_ids.return_value = {"chunk0"}
    embedding_repo.get_embedding_counts.return_value = 2 # Total is 2 at the end
    provider.embed_documents.return_value = [[0.4, 0.5, 0.6]] # Only 1 embedding returned

    job = RepresentationJob(job_id="job1", run_id=DUMMY_RUN_ID)

    with patch("app.ingestion.representation.service.BM25Index.save"):
        await service.run_representation_for_run(DUMMY_RUN_ID, job)

        # Provider only called with 1 chunk
        provider.embed_documents.assert_called_once()
        args, _ = provider.embed_documents.call_args
        assert len(args[0]) == 1
        assert args[0][0] == "This is chunk 1"
        
        # Only 1 embedding saved
        embedding_repo.save_embeddings.assert_called_once()
        args, _ = embedding_repo.save_embeddings.call_args
        assert len(args[0]) == 1
        assert args[0][0].chunk_id == "chunk1"

        assert job.embedded_chunks == 1 # 1 new embedding was done during this run
        assert job.total_chunks == 2


@pytest.mark.asyncio
async def test_run_representation_already_complete(service, mocks):
    provider, embedding_repo, paper_repo, storage_client = mocks
    
    # Simulate both chunks already embedded
    embedding_repo.get_embedded_chunk_ids.return_value = {"chunk0", "chunk1"}
    
    job = RepresentationJob(job_id="job1", run_id=DUMMY_RUN_ID)

    with patch("app.ingestion.representation.service.BM25Index.save"):
        await service.run_representation_for_run(DUMMY_RUN_ID, job)

        provider.embed_documents.assert_not_called()
        embedding_repo.save_embeddings.assert_not_called()
        
        assert job.embedded_chunks == 2 # Carried over from existing
        assert job.processed_papers == 1


@pytest.mark.asyncio
async def test_run_representation_no_papers(service, mocks):
    provider, embedding_repo, paper_repo, storage_client = mocks
    
    paper_repo.get_papers_for_representation.return_value = []
    
    job = RepresentationJob(job_id="job1", run_id=DUMMY_RUN_ID)
    await service.run_representation_for_run(DUMMY_RUN_ID, job)
    
    assert job.status == JobStatus.COMPLETED
    assert job.embedding_status == "NOT_EMBEDDED"
    assert job.bm25_status == "NOT_INDEXED"


@pytest.mark.asyncio
async def test_run_representation_partial_failure(service, mocks):
    provider, embedding_repo, paper_repo, storage_client = mocks
    
    provider.embed_documents.side_effect = Exception("API error")
    embedding_repo.get_embedding_counts.return_value = 0
    
    job = RepresentationJob(job_id="job1", run_id=DUMMY_RUN_ID)

    with patch("app.ingestion.representation.service.BM25Index.save"):
        await service.run_representation_for_run(DUMMY_RUN_ID, job)

        embedding_repo.save_embeddings.assert_not_called()
        paper_repo.update_embedding_status.assert_called_once_with(DUMMY_PAPER_ID, EmbeddingStatus.EMBEDDING_FAILED)
        
        assert job.status == JobStatus.COMPLETED
        assert job.embedding_status == "FAILED"
        assert job.failed_chunks == 2
        assert job.bm25_status == "INDEXED" # BM25 still builds from chunks even if embeddings fail


@pytest.mark.asyncio
async def test_run_representation_prunes_stale_embeddings(service, mocks):
    """After re-chunking, vectors of chunk_ids that no longer exist are deleted."""
    provider, embedding_repo, paper_repo, storage_client = mocks
    provider.get_model_name = MagicMock(return_value=DUMMY_MODEL)  # sync on the real provider
    embedding_repo.get_embedded_chunk_ids.return_value = {"chunk0", "old-a", "old-b"}
    provider.embed_documents.return_value = [[0.4, 0.5, 0.6]]  # only chunk1 is new
    job = RepresentationJob(job_id="job1", run_id=DUMMY_RUN_ID)

    with patch("app.ingestion.representation.service.BM25Index.save"):
        await service.run_representation_for_run(DUMMY_RUN_ID, job)

    embedding_repo.delete_embeddings.assert_awaited_once_with(["old-a", "old-b"], DUMMY_MODEL)
    args, _ = provider.embed_documents.call_args
    assert args[0] == ["This is chunk 1"]
