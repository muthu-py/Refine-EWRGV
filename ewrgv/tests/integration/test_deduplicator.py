"""
tests/integration/test_deduplicator.py
"""
import pytest
from unittest.mock import AsyncMock, MagicMock
import uuid

from app.ingestion.deduplication.service import Deduplicator, normalize_doi, normalize_title_authors_year

def test_normalize_doi():
    assert normalize_doi("https://doi.org/10.1234/ABC") == "10.1234/abc"
    assert normalize_doi("http://doi.org/10.1234/abc") == "10.1234/abc"
    assert normalize_doi("doi:10.1234/ABC ") == "10.1234/abc"
    assert normalize_doi("10.1234/ABC") == "10.1234/abc"
    assert normalize_doi(None) is None

def test_normalize_title_authors_year():
    assert normalize_title_authors_year("Graph Neural Networks", [{"name": "John Smith"}], 2024) == "graph neural networks|john_smith|2024"
    assert normalize_title_authors_year("Graph-Neural Networks!", [{"name": "Smith, John"}], 2024) == "graph neural networks|john_smith|2024"
    assert normalize_title_authors_year("Machine Learning", [], 2023) == "machine learning||2023"
    assert normalize_title_authors_year(None, [], 2023) is None
    assert normalize_title_authors_year("Machine Learning", [], None) is None

@pytest.fixture
def mock_db_pool():
    pool = MagicMock()
    conn = MagicMock()
    
    pool.acquire.return_value.__aenter__ = AsyncMock(return_value=conn)
    pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    
    conn.transaction.return_value.__aenter__ = AsyncMock(return_value=None)
    conn.transaction.return_value.__aexit__ = AsyncMock(return_value=None)
    
    return pool, conn

@pytest.mark.asyncio
async def test_deduplicator_run_merges_duplicates(mock_db_pool):
    pool, conn = mock_db_pool
    
    pid1 = str(uuid.uuid4())
    pid2 = str(uuid.uuid4())
    
    # Paper 1 has DOI, Paper 2 has same DOI but formatted differently
    papers = [
        {"paper_id": pid1, "title": "Paper X", "abstract": None, "doi": "10.1234/x", "publication_year": 2024, "authors": "[]"},
        {"paper_id": pid2, "title": "Paper X (duplicate)", "abstract": "Has abstract", "doi": "https://doi.org/10.1234/X", "publication_year": 2024, "authors": "[]"}
    ]
    
    sources = [
        {"source_id": "s1", "paper_id": pid1, "provider": "semantic_scholar", "provider_paper_id": "ss1"},
        {"source_id": "s2", "paper_id": pid2, "provider": "openalex", "provider_paper_id": "oa1"}
    ]
    
    async def mock_fetch(query, *args):
        if "FROM papers" in query:
            return papers
        elif "FROM paper_sources" in query:
            return sources
        return []

    conn.fetch = AsyncMock(side_effect=mock_fetch)
    conn.execute = AsyncMock(return_value=None)
    
    deduplicator = Deduplicator(pool)
    stats = await deduplicator.run()
    
    assert stats["components_merged"] == 1
    assert stats["duplicates_removed"] == 1
    
    # Verify execute calls inside transaction
    # 1. UPDATE papers
    # 2. UPDATE paper_sources
    # 3. DELETE FROM papers
    assert conn.execute.call_count == 3
    
    # Check UPDATE papers
    update_papers_call = conn.execute.call_args_list[0][0]
    assert "UPDATE papers" in update_papers_call[0]
    assert update_papers_call[2] == "Has abstract" # Abstract was merged from the paper that had it
    assert update_papers_call[3] is not None        # DOI is preserved (one of the two)
    
    # Check UPDATE paper_sources
    update_sources_call = conn.execute.call_args_list[1][0]
    assert "UPDATE paper_sources" in update_sources_call[0]
    
    # Check DELETE FROM papers
    delete_papers_call = conn.execute.call_args_list[2][0]
    assert "DELETE FROM papers" in delete_papers_call[0]

@pytest.mark.asyncio
async def test_deduplicator_run_no_duplicates(mock_db_pool):
    pool, conn = mock_db_pool
    
    papers = [
        {"paper_id": str(uuid.uuid4()), "title": "Paper A", "abstract": None, "doi": "10.1/a", "publication_year": 2024, "authors": "[]"},
        {"paper_id": str(uuid.uuid4()), "title": "Paper B", "abstract": None, "doi": "10.1/b", "publication_year": 2024, "authors": "[]"}
    ]
    sources = []
    
    async def mock_fetch(query, *args):
        if "FROM papers" in query:
            return papers
        return sources

    conn.fetch = AsyncMock(side_effect=mock_fetch)
    conn.execute = AsyncMock(return_value=None)
    
    deduplicator = Deduplicator(pool)
    stats = await deduplicator.run()
    
    assert stats["components_merged"] == 0
    assert stats["duplicates_removed"] == 0
    assert conn.execute.call_count == 0
