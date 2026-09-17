"""
tests/unit/test_parsed_document_model.py
-----------------------------------------
Unit tests for ParsedPage and ParsedDocument domain models (Phase 2.5).

Tests:
- Model instantiation with valid data
- Default field values
- ParsedPage.from_text() convenience constructor
- ParsedDocument.build() aggregate statistics
- JSON serialisation round-trip
- UTF-8 text survives serialisation
"""

from __future__ import annotations

import json

import pytest

from app.domain.models.parsed_document import ParsedDocument, ParsedPage


class TestParsedPage:
    def test_basic_instantiation(self):
        page = ParsedPage(page_number=1, text="Hello world", character_count=11)
        assert page.page_number == 1
        assert page.text == "Hello world"
        assert page.character_count == 11

    def test_from_text_computes_character_count(self):
        page = ParsedPage.from_text(page_number=3, text="abc")
        assert page.page_number == 3
        assert page.character_count == 3

    def test_from_text_empty(self):
        page = ParsedPage.from_text(page_number=1, text="")
        assert page.character_count == 0

    def test_from_text_unicode(self):
        text = "caf\u00e9"
        page = ParsedPage.from_text(page_number=1, text=text)
        assert page.character_count == 4  # Unicode chars, not bytes

    def test_one_based_page_number(self):
        """Page numbers must start at 1, not 0."""
        page = ParsedPage.from_text(page_number=1, text="first")
        assert page.page_number >= 1


class TestParsedDocument:
    def _make_pages(self, texts: list[str]) -> list[ParsedPage]:
        return [
            ParsedPage.from_text(page_number=i + 1, text=t)
            for i, t in enumerate(texts)
        ]

    def test_build_basic(self):
        pages = self._make_pages(["Hello", "World"])
        doc = ParsedDocument.build(
            paper_id="paper-123",
            source_storage_path="papers/paper-123/fulltext.pdf",
            pages=pages,
        )
        assert doc.paper_id == "paper-123"
        assert doc.page_count == 2
        assert doc.total_character_count == 10  # 5 + 5
        assert doc.pages_with_text == 2

    def test_build_empty_page_not_counted(self):
        pages = self._make_pages(["Hello", ""])
        doc = ParsedDocument.build(
            paper_id="paper-1",
            source_storage_path="papers/paper-1/fulltext.pdf",
            pages=pages,
        )
        assert doc.page_count == 2
        assert doc.pages_with_text == 1  # only the non-empty page
        assert doc.total_character_count == 5

    def test_build_whitespace_page_not_counted(self):
        pages = self._make_pages(["Text", "   \n  "])
        doc = ParsedDocument.build(
            paper_id="p", source_storage_path="s", pages=pages
        )
        assert doc.pages_with_text == 1

    def test_build_no_pages(self):
        doc = ParsedDocument.build(
            paper_id="empty", source_storage_path="papers/empty/fulltext.pdf", pages=[]
        )
        assert doc.page_count == 0
        assert doc.total_character_count == 0
        assert doc.pages_with_text == 0

    def test_build_with_metadata(self):
        doc = ParsedDocument.build(
            paper_id="p1",
            source_storage_path="s",
            pages=self._make_pages(["text"]),
            metadata={"title": "Test Paper", "creator": "LaTeX"},
        )
        assert doc.metadata["title"] == "Test Paper"

    def test_default_metadata_empty(self):
        doc = ParsedDocument.build(paper_id="p", source_storage_path="s", pages=[])
        assert doc.metadata == {}

    def test_json_serialisation_round_trip(self):
        pages = self._make_pages(["Page one text", "Page two text"])
        doc = ParsedDocument.build(
            paper_id="paper-json",
            source_storage_path="papers/paper-json/fulltext.pdf",
            pages=pages,
            metadata={"creator": "Word"},
        )
        json_str = doc.model_dump_json()
        data = json.loads(json_str)

        assert data["paper_id"] == "paper-json"
        assert data["page_count"] == 2
        assert len(data["pages"]) == 2
        assert data["pages"][0]["page_number"] == 1
        assert data["pages"][1]["page_number"] == 2
        assert data["metadata"]["creator"] == "Word"

    def test_utf8_text_survives_round_trip(self):
        text = "\u4e2d\u6587\u5185\u5bb9"  # Chinese characters
        pages = self._make_pages([text])
        doc = ParsedDocument.build(paper_id="p", source_storage_path="s", pages=pages)
        json_bytes = doc.model_dump_json().encode("utf-8")
        recovered = json.loads(json_bytes.decode("utf-8"))
        assert recovered["pages"][0]["text"] == text

    def test_page_ordering_preserved(self):
        pages = self._make_pages(["first", "second", "third"])
        doc = ParsedDocument.build(paper_id="p", source_storage_path="s", pages=pages)
        for i, page in enumerate(doc.pages):
            assert page.page_number == i + 1
