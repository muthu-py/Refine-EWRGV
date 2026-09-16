"""
tests/integration/test_supabase_persistence.py
----------------------------------------------
Tests for Supabase persistence layer mappings and error handling.
"""
import json
from unittest.mock import AsyncMock, MagicMock
import pytest

from app.domain.models.collection import CollectionResult, CollectedPaper, PaperProvenance
from app.domain.models.paper import Paper, Author
from app.storage.repositories.paper_repository import PaperRepository

@pytest.fixture
def mock_db_pool():
    pool = MagicMock()
    conn = MagicMock()
    
    # Setup async contexts
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    
    conn.transaction.return_value.__aenter__ = AsyncMock(return_value=None)
    conn.transaction.return_value.__aexit__ = AsyncMock(return_value=None)
    
    # Setup row returns
    conn.fetchrow = AsyncMock(return_value={"paper_id": "123e4567-e89b-12d3-a456-426614174000"})
    conn.execute = AsyncMock(return_value=None)
    
    return pool

@pytest.fixture
def sample_collection_result():
    paper1 = Paper(
        title="Test Paper 1",
        abstract="Test Abstract",
        doi="10.1234/test",
        publication_year=2024,
        authors=[Author(name="Alice")],
        paper_url="http://test.com",
        citation_count=5
    )
    prov1 = PaperProvenance(
        source="semantic_scholar",
        provider_id="ss123",
        source_query="test query"
    )
    
    paper2 = Paper(
        title="Test Paper 2",
        abstract="",
        doi=None,
        publication_year=None,
        authors=[],
        paper_url=None,
        citation_count=None
    )
    prov2 = PaperProvenance(
        source="openalex",
        provider_id="oa123",
        source_query="test query 2"
    )

    return CollectionResult(
        query_id="q1",
        queries_searched=["test query"],
        providers_used=["semantic_scholar"],
        papers=[
            CollectedPaper(paper=paper1, provenance=prov1),
            CollectedPaper(paper=paper1, provenance=prov2), # duplicate paper, diff prov
            CollectedPaper(paper=paper2, provenance=prov2)
        ],
        status="success"
    )

@pytest.mark.asyncio
async def test_save_collection_result_mappings(mock_db_pool, sample_collection_result):
    repo = PaperRepository(mock_db_pool)
    research_question = "How does RAG identify research gaps?"
    await repo.save_collection_result(sample_collection_result, research_question)
    
    conn = mock_db_pool.acquire.return_value.__aenter__.return_value
    
    # 1 execute for research_runs, 3 fetchrow for papers, 3 execute for paper_sources
    assert conn.execute.call_count == 4
    assert conn.fetchrow.call_count == 3
    
    # Check mapping for research_runs
    runs_execute_args = conn.execute.call_args_list[0][0]
    assert "INSERT INTO research_runs" in runs_execute_args[0]
    assert runs_execute_args[1] == "q1" # run_id (query_id)
    assert runs_execute_args[2] == research_question

    # Check mapping for first paper
    fetchrow_args = conn.fetchrow.call_args_list[0][0]
    assert fetchrow_args[1] == "Test Paper 1"
    assert fetchrow_args[2] == "Test Abstract"
    assert fetchrow_args[3] == "10.1234/test"
    assert fetchrow_args[4] == 2024
    assert "Alice" in fetchrow_args[5]  # authors JSON

    # Check mapping for provenance
    # It's the second execute call (index 1) because the first was research_runs
    sources_execute_args = conn.execute.call_args_list[1][0]
    assert sources_execute_args[1] == "123e4567-e89b-12d3-a456-426614174000" # paper_id
    assert sources_execute_args[2] == "q1" # run_id
    assert sources_execute_args[3] == "semantic_scholar" # provider
    assert sources_execute_args[4] == "ss123" # provider_id
    assert sources_execute_args[5] == "test query"
    assert sources_execute_args[6] == "http://test.com"
    assert sources_execute_args[7] == 5

@pytest.mark.asyncio
async def test_save_empty_collection(mock_db_pool):
    repo = PaperRepository(mock_db_pool)
    empty_result = CollectionResult(query_id="q1", papers=[])
    
    await repo.save_collection_result(empty_result, "Any query?")
    
    mock_db_pool.acquire.assert_not_called()
