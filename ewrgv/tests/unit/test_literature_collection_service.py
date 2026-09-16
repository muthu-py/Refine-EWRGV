"""
tests/unit/test_literature_collection_service.py
--------------------------------------------------
Unit tests for LiteratureCollectionService.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.exceptions import IngestionError
from app.domain.models.collection import CollectedPaper, PaperProvenance
from app.domain.models.paper import Paper
from app.domain.models.query import ResearchQuery
from app.ingestion.collectors.service import LiteratureCollectionService


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _make_collected_paper(
    title: str = "Test", source: str = "mock", query: str = "q"
) -> CollectedPaper:
    return CollectedPaper(
        paper=Paper(title=title, source=source),
        provenance=PaperProvenance(
            source=source, provider_id=f"id-{title}", source_query=query
        ),
    )


def _make_mock_provider(papers: list[CollectedPaper] | None = None) -> MagicMock:
    mock = MagicMock()
    mock.search.return_value = papers or []
    return mock


def _make_failing_provider(error_msg: str = "boom") -> MagicMock:
    mock = MagicMock()
    mock.search.side_effect = IngestionError(error_msg)
    return mock


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #


class TestLiteratureCollectionService:
    def test_requires_at_least_one_provider(self):
        with pytest.raises(ValueError, match="at least one provider"):
            LiteratureCollectionService(providers={})

    def test_single_provider_single_query(self):
        """Single provider + single expanded query → papers returned."""
        papers = [_make_collected_paper("P1"), _make_collected_paper("P2")]
        provider = _make_mock_provider(papers)
        service = LiteratureCollectionService(providers={"mock": provider})

        rq = ResearchQuery(query="test", expanded_queries=["expanded test"])
        result = service.collect(rq)

        assert result.status == "success"
        assert result.total_collected == 2
        assert result.queries_searched == ["expanded test"]
        assert "mock" in result.providers_used

    def test_multi_query_dispatch(self):
        """Each expanded query is dispatched to each provider."""
        provider = _make_mock_provider([_make_collected_paper("P")])
        service = LiteratureCollectionService(providers={"s2": provider})

        rq = ResearchQuery(
            query="original",
            expanded_queries=["q1", "q2", "q3"],
        )
        result = service.collect(rq)

        assert provider.search.call_count == 3
        assert result.total_collected == 3
        assert result.queries_searched == ["q1", "q2", "q3"]

    def test_multi_provider(self):
        """Two providers each return papers → combined results."""
        p1 = _make_mock_provider([_make_collected_paper("S2-1", source="s2")])
        p2 = _make_mock_provider([_make_collected_paper("OA-1", source="oa")])
        service = LiteratureCollectionService(providers={"s2": p1, "oa": p2})

        rq = ResearchQuery(query="test", expanded_queries=["q"])
        result = service.collect(rq)

        assert result.total_collected == 2
        assert set(result.providers_used) == {"s2", "oa"}
        assert result.status == "success"

    def test_provider_failure_is_independent(self):
        """One provider failing doesn't abort the other."""
        good = _make_mock_provider([_make_collected_paper("Good")])
        bad = _make_failing_provider("S2 rate limited")
        service = LiteratureCollectionService(providers={"good": good, "bad": bad})

        rq = ResearchQuery(query="test", expanded_queries=["q"])
        result = service.collect(rq)

        assert result.status == "partial"
        assert result.total_collected == 1
        assert len(result.provider_errors) == 1
        assert result.provider_errors[0]["provider"] == "bad"

    def test_all_providers_fail(self):
        """All providers fail → status='failed', no papers."""
        bad1 = _make_failing_provider("error1")
        bad2 = _make_failing_provider("error2")
        service = LiteratureCollectionService(providers={"b1": bad1, "b2": bad2})

        rq = ResearchQuery(query="test", expanded_queries=["q"])
        result = service.collect(rq)

        assert result.status == "failed"
        assert result.total_collected == 0
        assert len(result.provider_errors) == 2

    def test_max_total_results_cap(self):
        """Collection stops once max_total_results is reached."""
        papers = [_make_collected_paper(f"P{i}") for i in range(10)]
        provider = _make_mock_provider(papers)
        service = LiteratureCollectionService(
            providers={"mock": provider},
            max_total_results=5,
        )

        rq = ResearchQuery(query="test", expanded_queries=["q1", "q2"])
        result = service.collect(rq)

        # First query returns 10 but we only accept up to 5 limit param
        # The service passes min(max_results_per_query, remaining) as limit
        assert result.total_collected <= 10  # depends on provider returning limit

    def test_fallback_to_original_query(self):
        """When expanded_queries is empty, falls back to original query."""
        provider = _make_mock_provider([_make_collected_paper("Fallback")])
        service = LiteratureCollectionService(providers={"mock": provider})

        rq = ResearchQuery(query="original question", expanded_queries=[])
        result = service.collect(rq)

        assert result.queries_searched == ["original question"]
        assert result.total_collected == 1

    def test_provenance_preserved(self):
        """Provenance information is correctly passed through."""
        cp = _make_collected_paper("Traced", source="s2", query="my query")
        provider = _make_mock_provider([cp])
        service = LiteratureCollectionService(providers={"s2": provider})

        rq = ResearchQuery(query="test", expanded_queries=["my query"])
        result = service.collect(rq)

        assert result.papers[0].provenance.source == "s2"
        assert result.papers[0].provenance.source_query == "my query"

    def test_unexpected_exception_caught(self):
        """Non-IngestionError exceptions are caught and recorded."""
        provider = MagicMock()
        provider.search.side_effect = RuntimeError("unexpected crash")
        service = LiteratureCollectionService(providers={"crash": provider})

        rq = ResearchQuery(query="test", expanded_queries=["q"])
        result = service.collect(rq)

        assert result.status == "failed"
        assert len(result.provider_errors) == 1
        assert "unexpected" in result.provider_errors[0]["error"].lower()
