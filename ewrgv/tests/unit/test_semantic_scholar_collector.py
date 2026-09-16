"""
tests/unit/test_semantic_scholar_collector.py
----------------------------------------------
Unit tests for SemanticScholarCollector.

All HTTP calls are mocked — no real network traffic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.exceptions import IngestionError
from app.providers.search.semantic_scholar import SemanticScholarCollector


# ------------------------------------------------------------------ #
# Sample API responses
# ------------------------------------------------------------------ #

_SAMPLE_PAPER = {
    "paperId": "abc123",
    "title": "GNN-Based Intrusion Detection on CAN Bus",
    "authors": [
        {"name": "Alice Smith", "authorId": "1"},
        {"name": "Bob Jones", "authorId": "2"},
    ],
    "abstract": "We propose a graph neural network approach for CAN bus attacks.",
    "year": 2024,
    "venue": "NeurIPS",
    "externalIds": {"DOI": "10.1234/s2.2024.001", "ArXiv": "2401.12345"},
    "openAccessPdf": {"url": "https://arxiv.org/pdf/2401.12345", "status": "green"},
    "citationCount": 15,
    "url": "https://api.semanticscholar.org/paper/abc123",
}

_SAMPLE_RESPONSE = {"data": [_SAMPLE_PAPER], "total": 1, "offset": 0}

_EMPTY_RESPONSE = {"data": [], "total": 0, "offset": 0}


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #


class TestSemanticScholarCollector:
    def _make_collector(self, api_key: str = "") -> SemanticScholarCollector:
        return SemanticScholarCollector(api_key=api_key, timeout=5)

    @patch("app.providers.search.semantic_scholar.LiteratureHttpClient")
    def test_search_normalises_paper(self, MockClient):
        """A well-formed S2 response is normalised to CollectedPaper."""
        mock_http = MagicMock()
        mock_http.get.return_value = _SAMPLE_RESPONSE
        MockClient.return_value = mock_http

        collector = self._make_collector()
        collector._http = mock_http

        results = collector.search("GNN CAN bus", limit=10)
        assert len(results) == 1

        cp = results[0]
        assert cp.paper.title == "GNN-Based Intrusion Detection on CAN Bus"
        assert len(cp.paper.authors) == 2
        assert cp.paper.authors[0].name == "Alice Smith"
        assert cp.paper.abstract.startswith("We propose")
        assert cp.paper.doi == "10.1234/s2.2024.001"
        assert cp.paper.provider_id == "abc123"
        assert cp.paper.publication_year == 2024
        assert cp.paper.venue == "NeurIPS"
        assert cp.paper.citation_count == 15
        assert cp.paper.full_text_url == "https://arxiv.org/pdf/2401.12345"
        assert cp.paper.open_access_status == "green"
        assert cp.paper.source == "semantic_scholar"

        assert cp.provenance.source == "semantic_scholar"
        assert cp.provenance.provider_id == "abc123"
        assert cp.provenance.source_query == "GNN CAN bus"

    @patch("app.providers.search.semantic_scholar.LiteratureHttpClient")
    def test_search_empty_results(self, MockClient):
        """Empty data array → empty list, no error."""
        mock_http = MagicMock()
        mock_http.get.return_value = _EMPTY_RESPONSE
        MockClient.return_value = mock_http

        collector = self._make_collector()
        collector._http = mock_http
        results = collector.search("nonexistent topic")
        assert results == []

    @patch("app.providers.search.semantic_scholar.LiteratureHttpClient")
    def test_search_missing_fields_handled(self, MockClient):
        """Paper with minimal fields — missing DOI, abstract, venue."""
        minimal = {
            "paperId": "min1",
            "title": "Minimal",
            "authors": [],
            "abstract": None,
            "year": None,
            "venue": None,
            "externalIds": {},
            "openAccessPdf": None,
            "citationCount": None,
            "url": None,
        }
        mock_http = MagicMock()
        mock_http.get.return_value = {"data": [minimal]}
        MockClient.return_value = mock_http

        collector = self._make_collector()
        collector._http = mock_http
        results = collector.search("test")
        assert len(results) == 1
        cp = results[0]
        assert cp.paper.title == "Minimal"
        assert cp.paper.doi is None
        assert cp.paper.abstract == ""
        assert cp.paper.publication_year is None
        assert cp.paper.venue is None
        assert cp.paper.full_text_url is None

    @patch("app.providers.search.semantic_scholar.LiteratureHttpClient")
    def test_search_rate_limit_raises_ingestion_error(self, MockClient):
        """IngestionError from HTTP client propagates correctly."""
        mock_http = MagicMock()
        mock_http.get.side_effect = IngestionError("Rate limited (HTTP 429)")
        MockClient.return_value = mock_http

        collector = self._make_collector()
        collector._http = mock_http
        with pytest.raises(IngestionError, match="Rate limited"):
            collector.search("test")

    @patch("app.providers.search.semantic_scholar.LiteratureHttpClient")
    def test_search_malformed_response(self, MockClient):
        """Non-dict response raises IngestionError."""
        mock_http = MagicMock()
        mock_http.get.return_value = "not a dict"
        MockClient.return_value = mock_http

        collector = self._make_collector()
        collector._http = mock_http
        with pytest.raises(IngestionError, match="unexpected response"):
            collector.search("test")

    @patch("app.providers.search.semantic_scholar.LiteratureHttpClient")
    def test_search_skips_malformed_items(self, MockClient):
        """Malformed individual items are skipped, not fatal."""
        bad_item = "not a dict"  # will fail in _normalise
        mock_http = MagicMock()
        mock_http.get.return_value = {"data": [bad_item, _SAMPLE_PAPER]}
        MockClient.return_value = mock_http

        collector = self._make_collector()
        collector._http = mock_http
        results = collector.search("test")
        assert len(results) == 1  # only the valid paper

    @patch("app.providers.search.semantic_scholar.LiteratureHttpClient")
    def test_fetch_by_id_success(self, MockClient):
        """fetch_by_id returns a CollectedPaper for valid ID."""
        mock_http = MagicMock()
        mock_http.get.return_value = _SAMPLE_PAPER
        MockClient.return_value = mock_http

        collector = self._make_collector()
        collector._http = mock_http
        cp = collector.fetch_by_id("abc123")
        assert cp is not None
        assert cp.paper.provider_id == "abc123"

    @patch("app.providers.search.semantic_scholar.LiteratureHttpClient")
    def test_fetch_by_id_not_found(self, MockClient):
        """fetch_by_id returns None for 404."""
        mock_http = MagicMock()
        mock_http.get.side_effect = IngestionError("HTTP 404 from /paper/xyz")
        MockClient.return_value = mock_http

        collector = self._make_collector()
        collector._http = mock_http
        assert collector.fetch_by_id("xyz") is None

    def test_constructor_with_api_key(self):
        """When API key is provided, it should be configured."""
        collector = SemanticScholarCollector(api_key="test-key-123")
        assert collector._api_key == "test-key-123"

    def test_constructor_without_api_key(self):
        """Without API key, collector still initialises."""
        collector = SemanticScholarCollector()
        assert collector._api_key == ""
