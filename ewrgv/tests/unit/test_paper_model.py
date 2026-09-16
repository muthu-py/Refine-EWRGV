"""
tests/unit/test_paper_model.py
-------------------------------
Unit tests for the extended Paper model (Phase 2.2 fields).
"""

from __future__ import annotations

import pytest

from app.domain.models.paper import Author, Paper


class TestPaperLiteratureCollectionFields:
    """Verify the new optional fields added in Phase 2.2."""

    def test_paper_with_all_new_fields(self):
        """All Phase 2.2 fields can be set explicitly."""
        p = Paper(
            title="GNN for CAN Bus",
            doi="10.1234/test.2024",
            provider_id="abc123",
            paper_url="https://example.com/paper",
            full_text_url="https://example.com/paper.pdf",
            open_access_status="gold",
            citation_count=42,
            publication_year=2024,
        )
        assert p.doi == "10.1234/test.2024"
        assert p.provider_id == "abc123"
        assert p.paper_url == "https://example.com/paper"
        assert p.full_text_url == "https://example.com/paper.pdf"
        assert p.open_access_status == "gold"
        assert p.citation_count == 42
        assert p.publication_year == 2024

    def test_paper_defaults_none_for_new_fields(self):
        """All Phase 2.2 fields default to None when not provided."""
        p = Paper(title="Minimal Paper")
        assert p.doi is None
        assert p.provider_id is None
        assert p.paper_url is None
        assert p.full_text_url is None
        assert p.open_access_status is None
        assert p.citation_count is None
        assert p.publication_year is None

    def test_paper_backward_compat(self):
        """Existing fields still work as before (no regressions)."""
        p = Paper(
            title="Test",
            authors=[Author(name="Alice")],
            abstract="An abstract.",
            source="semantic_scholar",
            venue="NeurIPS",
        )
        assert p.title == "Test"
        assert len(p.authors) == 1
        assert p.authors[0].name == "Alice"
        assert p.abstract == "An abstract."
        assert p.source == "semantic_scholar"
        assert p.venue == "NeurIPS"
        assert p.sections == {}
        assert p.metadata == {}

    def test_paper_serialization_round_trip(self):
        """Paper with new fields can be serialized and deserialized."""
        p = Paper(
            title="Round Trip",
            doi="10.5678/rt",
            citation_count=7,
            publication_year=2023,
        )
        d = p.model_dump()
        p2 = Paper.model_validate(d)
        assert p2.doi == "10.5678/rt"
        assert p2.citation_count == 7
        assert p2.publication_year == 2023
