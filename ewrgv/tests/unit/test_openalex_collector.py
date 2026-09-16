"""
tests/unit/test_openalex_collector.py
---------------------------------------
Unit tests for OpenAlexCollector.

All HTTP calls are mocked — no real network traffic.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.exceptions import IngestionError
from app.providers.search.openalex import OpenAlexCollector, _reconstruct_abstract


# ------------------------------------------------------------------ #
# Sample API responses
# ------------------------------------------------------------------ #

_SAMPLE_WORK = {
    "id": "https://openalex.org/W1234567890",
    "title": "Deep Learning for Autonomous Driving",
    "authorships": [
        {
            "author": {"display_name": "Carol Lee", "id": "A1"},
            "institutions": [{"display_name": "MIT"}],
        },
        {
            "author": {"display_name": "Dave Kim", "id": "A2"},
            "institutions": [],
        },
    ],
    "abstract_inverted_index": {
        "We": [0],
        "present": [1],
        "a": [2],
        "deep": [3],
        "learning": [4],
        "approach.": [5],
    },
    "publication_year": 2023,
    "primary_location": {
        "source": {"display_name": "Nature Machine Intelligence", "id": "S1"},
    },
    "doi": "https://doi.org/10.1038/s42256-023-001",
    "open_access": {"is_oa": True, "oa_status": "gold", "oa_url": "https://example.com/paper.pdf"},
    "cited_by_count": 128,
    "type": "journal-article",
}

_SAMPLE_RESPONSE = {
    "meta": {"count": 1, "page": 1, "per_page": 25},
    "results": [_SAMPLE_WORK],
}

_EMPTY_RESPONSE = {"meta": {"count": 0}, "results": []}


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #


class TestReconstructAbstract:
    def test_normal_index(self):
        idx = {"Hello": [0], "world": [1]}
        assert _reconstruct_abstract(idx) == "Hello world"

    def test_multi_position(self):
        idx = {"the": [0, 3], "cat": [1], "sat": [2], "on": [4], "mat": [5]}
        assert _reconstruct_abstract(idx) == "the cat sat the on mat"

    def test_none_returns_empty(self):
        assert _reconstruct_abstract(None) == ""

    def test_empty_dict_returns_empty(self):
        assert _reconstruct_abstract({}) == ""


class TestOpenAlexCollector:
    def _make_collector(self) -> OpenAlexCollector:
        return OpenAlexCollector(mailto="test@example.com", timeout=5)

    @patch("app.providers.search.openalex.LiteratureHttpClient")
    def test_search_normalises_work(self, MockClient):
        """A well-formed OpenAlex response is normalised to CollectedPaper."""
        mock_http = MagicMock()
        mock_http.get.return_value = _SAMPLE_RESPONSE
        MockClient.return_value = mock_http

        collector = self._make_collector()
        collector._http = mock_http
        results = collector.search("deep learning autonomous", limit=10)

        assert len(results) == 1
        cp = results[0]
        assert cp.paper.title == "Deep Learning for Autonomous Driving"
        assert len(cp.paper.authors) == 2
        assert cp.paper.authors[0].name == "Carol Lee"
        assert cp.paper.abstract == "We present a deep learning approach."
        assert cp.paper.doi == "10.1038/s42256-023-001"
        assert cp.paper.provider_id == "W1234567890"
        assert cp.paper.publication_year == 2023
        assert cp.paper.venue == "Nature Machine Intelligence"
        assert cp.paper.citation_count == 128
        assert cp.paper.open_access_status == "gold"
        assert cp.paper.full_text_url == "https://example.com/paper.pdf"
        assert cp.paper.source == "openalex"

        assert cp.provenance.source == "openalex"
        assert cp.provenance.provider_id == "W1234567890"
        assert cp.provenance.source_query == "deep learning autonomous"

    @patch("app.providers.search.openalex.LiteratureHttpClient")
    def test_search_empty_results(self, MockClient):
        """Empty results → empty list."""
        mock_http = MagicMock()
        mock_http.get.return_value = _EMPTY_RESPONSE
        MockClient.return_value = mock_http

        collector = self._make_collector()
        collector._http = mock_http
        results = collector.search("nonexistent")
        assert results == []

    @patch("app.providers.search.openalex.LiteratureHttpClient")
    def test_search_missing_fields(self, MockClient):
        """Work with minimal fields — no DOI, no abstract, no venue."""
        minimal = {
            "id": "https://openalex.org/W999",
            "title": "Minimal Work",
            "authorships": [],
            "abstract_inverted_index": None,
            "publication_year": None,
            "primary_location": None,
            "doi": None,
            "open_access": None,
            "cited_by_count": None,
        }
        mock_http = MagicMock()
        mock_http.get.return_value = {"results": [minimal]}
        MockClient.return_value = mock_http

        collector = self._make_collector()
        collector._http = mock_http
        results = collector.search("test")
        assert len(results) == 1
        cp = results[0]
        assert cp.paper.title == "Minimal Work"
        assert cp.paper.doi is None
        assert cp.paper.abstract == ""
        assert cp.paper.venue is None
        assert cp.paper.publication_year is None

    @patch("app.providers.search.openalex.LiteratureHttpClient")
    def test_search_rate_limit_raises_ingestion_error(self, MockClient):
        """IngestionError propagates."""
        mock_http = MagicMock()
        mock_http.get.side_effect = IngestionError("Rate limited")
        MockClient.return_value = mock_http

        collector = self._make_collector()
        collector._http = mock_http
        with pytest.raises(IngestionError):
            collector.search("test")

    @patch("app.providers.search.openalex.LiteratureHttpClient")
    def test_search_malformed_response(self, MockClient):
        """Non-dict response raises IngestionError."""
        mock_http = MagicMock()
        mock_http.get.return_value = "bad"
        MockClient.return_value = mock_http

        collector = self._make_collector()
        collector._http = mock_http
        with pytest.raises(IngestionError, match="unexpected response"):
            collector.search("test")

    @patch("app.providers.search.openalex.LiteratureHttpClient")
    def test_doi_stripping(self, MockClient):
        """DOI has 'https://doi.org/' prefix stripped."""
        work = dict(_SAMPLE_WORK)
        work["doi"] = "https://doi.org/10.9999/test"
        mock_http = MagicMock()
        mock_http.get.return_value = {"results": [work]}
        MockClient.return_value = mock_http

        collector = self._make_collector()
        collector._http = mock_http
        results = collector.search("test")
        assert results[0].paper.doi == "10.9999/test"

    @patch("app.providers.search.openalex.LiteratureHttpClient")
    def test_fetch_by_id_success(self, MockClient):
        """fetch_by_id returns CollectedPaper."""
        mock_http = MagicMock()
        mock_http.get.return_value = _SAMPLE_WORK
        MockClient.return_value = mock_http

        collector = self._make_collector()
        collector._http = mock_http
        cp = collector.fetch_by_id("W1234567890")
        assert cp is not None
        assert cp.paper.provider_id == "W1234567890"

    @patch("app.providers.search.openalex.LiteratureHttpClient")
    def test_fetch_by_id_not_found(self, MockClient):
        """fetch_by_id returns None for 404."""
        mock_http = MagicMock()
        mock_http.get.side_effect = IngestionError("HTTP 404 from /works/xxx")
        MockClient.return_value = mock_http

        collector = self._make_collector()
        collector._http = mock_http
        assert collector.fetch_by_id("xxx") is None
