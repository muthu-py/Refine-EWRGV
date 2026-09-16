"""
tests/integration/test_literature_collection_integration.py
-------------------------------------------------------------
Integration tests for the full literature collection pipeline.

Uses mocked HTTP responses (no real network calls) but exercises
the full path:  ResearchQuery → LiteratureCollectionService
  → SemanticScholarCollector + OpenAlexCollector → CollectionResult
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.domain.models.query import ResearchQuery
from app.ingestion.collectors.service import LiteratureCollectionService
from app.providers.search.openalex import OpenAlexCollector
from app.providers.search.semantic_scholar import SemanticScholarCollector


# ------------------------------------------------------------------ #
# Sample responses
# ------------------------------------------------------------------ #

_S2_RESPONSE = {
    "data": [
        {
            "paperId": "s2-001",
            "title": "S2 Paper One",
            "authors": [{"name": "Author A"}],
            "abstract": "Abstract from S2.",
            "year": 2024,
            "venue": "ICML",
            "externalIds": {"DOI": "10.1111/s2.001"},
            "openAccessPdf": None,
            "citationCount": 5,
            "url": "https://s2.org/paper/s2-001",
        }
    ]
}

_OA_RESPONSE = {
    "results": [
        {
            "id": "https://openalex.org/W9999",
            "title": "OA Paper One",
            "authorships": [
                {"author": {"display_name": "Author B"}, "institutions": []},
            ],
            "abstract_inverted_index": {"Hello": [0], "world": [1]},
            "publication_year": 2023,
            "primary_location": {"source": {"display_name": "Nature"}},
            "doi": "https://doi.org/10.2222/oa.001",
            "open_access": {"oa_status": "gold", "oa_url": "https://oa.com/pdf"},
            "cited_by_count": 99,
        }
    ]
}


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #


class TestLiteratureCollectionIntegration:
    def test_end_to_end_with_both_providers(self):
        """Full pipeline: ResearchQuery → both providers → CollectionResult."""
        # Create collectors with real classes but mock their HTTP clients
        s2 = SemanticScholarCollector()
        s2._http = MagicMock()
        s2._http.get.return_value = _S2_RESPONSE

        oa = OpenAlexCollector(mailto="test@test.com")
        oa._http = MagicMock()
        oa._http.get.return_value = _OA_RESPONSE

        service = LiteratureCollectionService(
            providers={"semantic_scholar": s2, "openalex": oa},
            max_results_per_query=20,
            max_total_results=100,
        )

        rq = ResearchQuery(
            query="graph neural networks CAN bus",
            expanded_queries=["GNN intrusion detection"],
        )
        result = service.collect(rq)

        assert result.status == "success"
        assert result.total_collected == 2
        assert set(result.providers_used) == {"semantic_scholar", "openalex"}
        assert result.queries_searched == ["GNN intrusion detection"]
        assert len(result.provider_errors) == 0

        # Verify S2 paper
        s2_papers = [cp for cp in result.papers if cp.provenance.source == "semantic_scholar"]
        assert len(s2_papers) == 1
        assert s2_papers[0].paper.title == "S2 Paper One"
        assert s2_papers[0].paper.doi == "10.1111/s2.001"
        assert s2_papers[0].paper.source == "semantic_scholar"

        # Verify OA paper
        oa_papers = [cp for cp in result.papers if cp.provenance.source == "openalex"]
        assert len(oa_papers) == 1
        assert oa_papers[0].paper.title == "OA Paper One"
        assert oa_papers[0].paper.doi == "10.2222/oa.001"
        assert oa_papers[0].paper.source == "openalex"
        assert oa_papers[0].paper.abstract == "Hello world"

    def test_multi_query_both_providers(self):
        """Multiple expanded queries dispatched to both providers."""
        s2 = SemanticScholarCollector()
        s2._http = MagicMock()
        s2._http.get.return_value = _S2_RESPONSE

        oa = OpenAlexCollector()
        oa._http = MagicMock()
        oa._http.get.return_value = _OA_RESPONSE

        service = LiteratureCollectionService(
            providers={"semantic_scholar": s2, "openalex": oa},
        )

        rq = ResearchQuery(
            query="base query",
            expanded_queries=["q1", "q2"],
        )
        result = service.collect(rq)

        # 2 queries × 2 providers × 1 paper each = 4 total
        assert result.total_collected == 4
        assert s2._http.get.call_count == 2
        assert oa._http.get.call_count == 2

    def test_one_provider_fails(self):
        """One provider fails; the other still contributes papers."""
        from app.core.exceptions import IngestionError

        s2 = SemanticScholarCollector()
        s2._http = MagicMock()
        s2._http.get.side_effect = IngestionError("S2 down")

        oa = OpenAlexCollector()
        oa._http = MagicMock()
        oa._http.get.return_value = _OA_RESPONSE

        service = LiteratureCollectionService(
            providers={"semantic_scholar": s2, "openalex": oa},
        )

        rq = ResearchQuery(query="test", expanded_queries=["q"])
        result = service.collect(rq)

        assert result.status == "partial"
        assert result.total_collected == 1
        assert "openalex" in result.providers_used
        assert len(result.provider_errors) == 1
