"""
tests/unit/test_collection_models.py
--------------------------------------
Unit tests for PaperProvenance, CollectedPaper, and CollectionResult.
"""

from __future__ import annotations

import pytest

from app.domain.models.collection import (
    CollectedPaper,
    CollectionResult,
    PaperProvenance,
)
from app.domain.models.paper import Author, Paper


def _make_paper(**overrides) -> Paper:
    defaults = {"title": "Test Paper", "source": "semantic_scholar"}
    defaults.update(overrides)
    return Paper(**defaults)


class TestPaperProvenance:
    def test_construction(self):
        prov = PaperProvenance(
            source="semantic_scholar",
            provider_id="abc123",
            source_query="graph neural network CAN bus",
        )
        assert prov.source == "semantic_scholar"
        assert prov.provider_id == "abc123"
        assert prov.source_query == "graph neural network CAN bus"


class TestCollectedPaper:
    def test_construction(self):
        paper = _make_paper()
        prov = PaperProvenance(
            source="openalex", provider_id="W123", source_query="test"
        )
        cp = CollectedPaper(paper=paper, provenance=prov)
        assert cp.paper.title == "Test Paper"
        assert cp.provenance.source == "openalex"


class TestCollectionResult:
    def test_empty_result(self):
        r = CollectionResult(query_id="q1")
        assert r.query_id == "q1"
        assert r.papers == []
        assert r.total_collected == 0
        assert r.status == "success"

    def test_total_collected_property(self):
        papers = [
            CollectedPaper(
                paper=_make_paper(title=f"Paper {i}"),
                provenance=PaperProvenance(
                    source="s2", provider_id=f"id{i}", source_query="q"
                ),
            )
            for i in range(3)
        ]
        r = CollectionResult(query_id="q2", papers=papers)
        assert r.total_collected == 3

    def test_status_values(self):
        for status in ["success", "partial", "failed"]:
            r = CollectionResult(query_id="q", status=status)
            assert r.status == status

    def test_provider_errors_populated(self):
        r = CollectionResult(
            query_id="q3",
            provider_errors=[
                {"provider": "semantic_scholar", "query": "test", "error": "timeout"},
            ],
            status="partial",
        )
        assert len(r.provider_errors) == 1
        assert r.provider_errors[0]["provider"] == "semantic_scholar"

    def test_serialization_round_trip(self):
        cp = CollectedPaper(
            paper=_make_paper(doi="10.1234/x"),
            provenance=PaperProvenance(
                source="openalex", provider_id="W999", source_query="test query"
            ),
        )
        r = CollectionResult(
            query_id="q4",
            queries_searched=["test query"],
            providers_used=["openalex"],
            papers=[cp],
            status="success",
        )
        d = r.model_dump()
        r2 = CollectionResult.model_validate(d)
        assert r2.total_collected == 1
        assert r2.papers[0].paper.doi == "10.1234/x"
