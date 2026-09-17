"""
tests/integration/test_supabase_persistence.py
----------------------------------------------
Tests for Supabase persistence layer mappings and error handling.

Phase 2.8 additions:
- Cross-run paper reuse
- Provenance preservation (multiple providers, queries, runs)
- Corpus health count queries
- Corpus consistency check fetch
- get_paper_by_id
- Status-consistency: artifact must exist before success status
- Security: no secrets logged
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from app.domain.models.collection import CollectionResult, CollectedPaper, PaperProvenance
from app.domain.models.paper import Paper, Author
from app.storage.repositories.paper_repository import PaperRepository


# ------------------------------------------------------------------ #
# Shared fixtures
# ------------------------------------------------------------------ #


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
            CollectedPaper(paper=paper1, provenance=prov2),  # duplicate paper, diff prov
            CollectedPaper(paper=paper2, provenance=prov2)
        ],
        status="success"
    )


# ------------------------------------------------------------------ #
# Existing tests (preserved, unchanged)
# ------------------------------------------------------------------ #


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
    assert runs_execute_args[1] == "q1"  # run_id (query_id)
    assert runs_execute_args[2] == research_question

    # Check mapping for first paper
    fetchrow_args = conn.fetchrow.call_args_list[0][0]
    assert fetchrow_args[1] == "Test Paper 1"
    assert fetchrow_args[2] == "Test Abstract"
    assert fetchrow_args[3] == "10.1234/test"
    assert fetchrow_args[4] == 2024
    assert "Alice" in fetchrow_args[5]  # authors JSON

    # Check mapping for provenance
    sources_execute_args = conn.execute.call_args_list[1][0]
    assert sources_execute_args[1] == "123e4567-e89b-12d3-a456-426614174000"  # paper_id
    assert sources_execute_args[2] == "q1"  # run_id
    assert sources_execute_args[3] == "semantic_scholar"  # provider
    assert sources_execute_args[4] == "ss123"  # provider_id
    assert sources_execute_args[5] == "test query"
    assert sources_execute_args[6] == "http://test.com"
    assert sources_execute_args[7] == 5


@pytest.mark.asyncio
async def test_save_empty_collection(mock_db_pool):
    repo = PaperRepository(mock_db_pool)
    empty_result = CollectionResult(query_id="q1", papers=[])

    await repo.save_collection_result(empty_result, "Any query?")

    mock_db_pool.acquire.assert_not_called()


# ------------------------------------------------------------------ #
# Phase 2.8: Canonical paper persistence
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_doi_based_upsert_uses_on_conflict(mock_db_pool):
    """
    Verify the upsert SQL contains ON CONFLICT (doi) DO UPDATE,
    ensuring canonical paper deduplication by DOI.
    """
    conn = mock_db_pool.acquire.return_value.__aenter__.return_value
    conn.fetchrow = AsyncMock(return_value={"paper_id": "paper-uuid-1"})

    paper = Paper(
        title="DOI Paper",
        doi="10.9999/unique-doi",
        authors=[Author(name="Bob")],
        publication_year=2023,
    )
    prov = PaperProvenance(source="openalex", provider_id="oa-456", source_query="q")
    result = CollectionResult(
        query_id="run-x",
        papers=[CollectedPaper(paper=paper, provenance=prov)],
    )

    repo = PaperRepository(mock_db_pool)
    await repo.save_collection_result(result, "Test question")

    # Check the fetchrow (paper upsert) SQL
    fetchrow_calls = conn.fetchrow.call_args_list
    assert len(fetchrow_calls) == 1
    sql = fetchrow_calls[0][0][0]
    assert "ON CONFLICT" in sql
    assert "doi" in sql.lower()


# ------------------------------------------------------------------ #
# Phase 2.8: Cross-run paper reuse
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_cross_run_reuse_same_paper_id(mock_db_pool):
    """
    When the same paper (by DOI) is discovered in two different runs,
    the upsert must reuse the same paper_id (canonical).
    The paper_sources table gets a new row for each run.
    """
    canonical_paper_id = "canonical-uuid-abc"
    conn = mock_db_pool.acquire.return_value.__aenter__.return_value
    conn.fetchrow = AsyncMock(return_value={"paper_id": canonical_paper_id})

    paper = Paper(
        title="Shared Paper",
        doi="10.8888/shared",
        authors=[Author(name="Carol")],
        publication_year=2022,
    )

    # Run A
    prov_a = PaperProvenance(source="semantic_scholar", provider_id="s2-111", source_query="q_a")
    result_a = CollectionResult(
        query_id="run-A",
        papers=[CollectedPaper(paper=paper, provenance=prov_a)],
    )

    # Run B — discovers the same paper
    prov_b = PaperProvenance(source="openalex", provider_id="oa-222", source_query="q_b")
    result_b = CollectionResult(
        query_id="run-B",
        papers=[CollectedPaper(paper=paper, provenance=prov_b)],
    )

    repo = PaperRepository(mock_db_pool)
    await repo.save_collection_result(result_a, "Question A")
    await repo.save_collection_result(result_b, "Question B")

    # Fetchrow was called twice (once per save), returning same canonical_paper_id
    assert conn.fetchrow.call_count == 2

    # Both provenance insertions should use the same paper_id
    paper_sources_executes = [
        call for call in conn.execute.call_args_list
        if "INSERT INTO paper_sources" in call[0][0]
    ]
    assert len(paper_sources_executes) == 2
    for call in paper_sources_executes:
        assert call[0][1] == canonical_paper_id  # same canonical paper_id used


# ------------------------------------------------------------------ #
# Phase 2.8: Provenance preservation
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_provenance_preserved_multiple_providers(mock_db_pool):
    """
    Same paper discovered by Semantic Scholar AND OpenAlex in the same run →
    two paper_sources rows, each with correct provider info.
    """
    canonical_paper_id = "prov-test-uuid"
    conn = mock_db_pool.acquire.return_value.__aenter__.return_value
    conn.fetchrow = AsyncMock(return_value={"paper_id": canonical_paper_id})

    paper = Paper(
        title="Multi-Provider Paper",
        doi="10.7777/multi",
        authors=[Author(name="Dave")],
        publication_year=2021,
    )
    prov_ss = PaperProvenance(source="semantic_scholar", provider_id="ss-001", source_query="q1")
    prov_oa = PaperProvenance(source="openalex", provider_id="oa-001", source_query="q1")

    result = CollectionResult(
        query_id="run-multi",
        papers=[
            CollectedPaper(paper=paper, provenance=prov_ss),
            CollectedPaper(paper=paper, provenance=prov_oa),
        ],
    )

    repo = PaperRepository(mock_db_pool)
    await repo.save_collection_result(result, "Multi-provider question")

    sources_inserts = [
        call for call in conn.execute.call_args_list
        if "INSERT INTO paper_sources" in call[0][0]
    ]
    assert len(sources_inserts) == 2

    providers = {call[0][3] for call in sources_inserts}  # index 3 = provider
    assert "semantic_scholar" in providers
    assert "openalex" in providers


@pytest.mark.asyncio
async def test_provenance_all_fields_preserved(mock_db_pool):
    """
    Verify all provenance fields are correctly passed to paper_sources.
    """
    paper_id = "full-prov-uuid"
    conn = mock_db_pool.acquire.return_value.__aenter__.return_value
    conn.fetchrow = AsyncMock(return_value={"paper_id": paper_id})

    paper = Paper(
        title="Provenance Test Paper",
        doi="10.6666/prov",
        authors=[Author(name="Eve")],
        publication_year=2020,
        paper_url="https://doi.org/10.6666/prov",
        citation_count=42,
    )
    prov = PaperProvenance(
        source="semantic_scholar",
        provider_id="ss-prov-test",
        source_query="my expanded query string"
    )
    result = CollectionResult(
        query_id="run-prov",
        papers=[CollectedPaper(paper=paper, provenance=prov)],
    )

    repo = PaperRepository(mock_db_pool)
    await repo.save_collection_result(result, "Provenance question")

    sources_calls = [
        call for call in conn.execute.call_args_list
        if "INSERT INTO paper_sources" in call[0][0]
    ]
    assert len(sources_calls) == 1
    args = sources_calls[0][0]
    # args: (sql, paper_id, run_id, provider, provider_paper_id, source_query, paper_url, citation_count)
    assert args[1] == paper_id                         # paper_id
    assert args[2] == "run-prov"                       # run_id
    assert args[3] == "semantic_scholar"               # provider
    assert args[4] == "ss-prov-test"                   # provider_paper_id
    assert args[5] == "my expanded query string"       # source_query
    assert args[6] == "https://doi.org/10.6666/prov"   # paper_url
    assert args[7] == 42                               # citation_count


# ------------------------------------------------------------------ #
# Phase 2.8: Corpus health query methods
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_get_corpus_health_counts_global(mock_db_pool):
    """get_corpus_health_counts() with no run_id executes the global count query."""
    conn = mock_db_pool.acquire.return_value.__aenter__.return_value
    conn.fetchrow = AsyncMock(return_value={
        "total_papers": 100,
        "downloaded": 30,
        "not_checked": 50,
        "no_oa_found": 10,
        "download_failed": 5,
        "invalid_document": 5,
        "parsed": 25,
        "sections_extracted": 20,
        "chunked": 15,
    })

    repo = PaperRepository(mock_db_pool)
    counts = await repo.get_corpus_health_counts(run_id=None)

    assert counts["total_papers"] == 100
    assert counts["downloaded"] == 30
    assert counts["parsed"] == 25
    assert counts["chunked"] == 15

    # Should NOT join paper_sources (no run_id filter)
    sql = conn.fetchrow.call_args[0][0]
    assert "paper_sources" not in sql.lower()


@pytest.mark.asyncio
async def test_get_corpus_health_counts_by_run(mock_db_pool):
    """get_corpus_health_counts() with run_id joins paper_sources."""
    conn = mock_db_pool.acquire.return_value.__aenter__.return_value
    conn.fetchrow = AsyncMock(return_value={
        "total_papers": 10, "downloaded": 5,
        "not_checked": 3, "no_oa_found": 2, "download_failed": 0,
        "invalid_document": 0, "parsed": 4,
        "sections_extracted": 3, "chunked": 2,
    })

    repo = PaperRepository(mock_db_pool)
    counts = await repo.get_corpus_health_counts(run_id="specific-run-id")

    assert counts["total_papers"] == 10

    # Should join paper_sources for the run filter
    sql, run_id_arg = conn.fetchrow.call_args[0]
    assert "paper_sources" in sql.lower()
    assert run_id_arg == "specific-run-id"


@pytest.mark.asyncio
async def test_get_papers_for_consistency_check(mock_db_pool):
    """get_papers_for_consistency_check() returns list of dicts with status fields."""
    conn = mock_db_pool.acquire.return_value.__aenter__.return_value
    conn.fetch = AsyncMock(return_value=[
        {
            "paper_id": "p1",
            "full_text_status": "DOWNLOADED",
            "storage_path": "papers/p1/fulltext.pdf",
            "parsing_status": "PARSED",
            "parsed_storage_path": "papers/p1/parsed.json",
            "section_status": "NOT_EXTRACTED",
            "sections_storage_path": None,
            "chunk_status": "NOT_CHUNKED",
            "chunks_storage_path": None,
        }
    ])

    repo = PaperRepository(mock_db_pool)
    papers = await repo.get_papers_for_consistency_check(run_id=None)

    assert len(papers) == 1
    assert papers[0]["paper_id"] == "p1"
    assert papers[0]["parsing_status"] == "PARSED"
    assert papers[0]["chunk_status"] == "NOT_CHUNKED"


@pytest.mark.asyncio
async def test_get_provenance_for_paper(mock_db_pool):
    """get_provenance_for_paper() returns all paper_sources rows for a paper."""
    conn = mock_db_pool.acquire.return_value.__aenter__.return_value
    conn.fetch = AsyncMock(return_value=[
        {
            "source_id": "src-1", "paper_id": "pA", "run_id": "run-1",
            "provider": "semantic_scholar", "provider_paper_id": "s2-001",
            "source_query": "q1", "paper_url": None, "citation_count": 10,
            "created_at": None
        },
        {
            "source_id": "src-2", "paper_id": "pA", "run_id": "run-2",
            "provider": "openalex", "provider_paper_id": "oa-001",
            "source_query": "q2", "paper_url": None, "citation_count": 10,
            "created_at": None
        },
    ])

    repo = PaperRepository(mock_db_pool)
    provenance = await repo.get_provenance_for_paper("pA")

    assert len(provenance) == 2
    providers = {p["provider"] for p in provenance}
    assert "semantic_scholar" in providers
    assert "openalex" in providers


# ------------------------------------------------------------------ #
# Phase 2.8: run_exists
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_run_exists_true(mock_db_pool):
    conn = mock_db_pool.acquire.return_value.__aenter__.return_value
    conn.fetchval = AsyncMock(return_value=1)

    repo = PaperRepository(mock_db_pool)
    assert await repo.run_exists("existing-run") is True


@pytest.mark.asyncio
async def test_run_exists_false(mock_db_pool):
    conn = mock_db_pool.acquire.return_value.__aenter__.return_value
    conn.fetchval = AsyncMock(return_value=None)

    repo = PaperRepository(mock_db_pool)
    assert await repo.run_exists("non-existent-run") is False


# ------------------------------------------------------------------ #
# Phase 2.8: Status consistency — artifact must exist before success status
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_update_parsing_metadata_only_sets_path_on_success(mock_db_pool):
    """
    update_parsing_metadata must only be called AFTER a successful upload.
    The test verifies that when called with status=PARSED, the storage path
    is included in the UPDATE.
    """
    conn = mock_db_pool.acquire.return_value.__aenter__.return_value

    repo = PaperRepository(mock_db_pool)
    await repo.update_parsing_metadata(
        paper_id="p1",
        status="PARSED",
        parsed_storage_path="papers/p1/parsed.json",
        page_count=10,
    )

    sql, status, path, pages, pid = conn.execute.call_args[0]
    assert status == "PARSED"
    assert path == "papers/p1/parsed.json"
    assert pages == 10
    assert pid == "p1"
    assert "parsing_status" in sql


@pytest.mark.asyncio
async def test_update_parsing_metadata_failure_sets_no_path(mock_db_pool):
    """
    When status=PARSE_FAILED, no storage path should be provided.
    """
    conn = mock_db_pool.acquire.return_value.__aenter__.return_value

    repo = PaperRepository(mock_db_pool)
    await repo.update_parsing_metadata(
        paper_id="p2",
        status="PARSE_FAILED",
        parsed_storage_path=None,
    )

    sql, status, path, pages, pid = conn.execute.call_args[0]
    assert status == "PARSE_FAILED"
    assert path is None


# ------------------------------------------------------------------ #
# Security: no secrets logged
# ------------------------------------------------------------------ #


def test_paper_repository_does_not_log_dsn(caplog, mock_db_pool):
    """PaperRepository construction must not log database credentials."""
    import logging
    with caplog.at_level(logging.DEBUG):
        PaperRepository(mock_db_pool)

    for record in caplog.records:
        msg = record.getMessage()
        assert "password" not in msg.lower()
        assert "postgresql://" not in msg.lower()

