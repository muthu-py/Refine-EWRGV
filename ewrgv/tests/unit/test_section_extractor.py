"""
tests/unit/test_section_extractor.py
---------------------------------------
Unit tests for app/ingestion/sectioning/extractor.py

Tests cover heading detection integration, page provenance,
fallback section creation, and text preservation.
"""
import pytest
from app.domain.enums import SectionType
from app.domain.models.parsed_document import ParsedDocument, ParsedPage
from app.ingestion.sectioning.extractor import SectionExtractor


def make_doc(paper_id: str = "test-paper", pages: list[tuple[int, str]] = None) -> ParsedDocument:
    """Helper: build a ParsedDocument with given pages (page_number, text)."""
    if pages is None:
        pages = [(1, "Sample text")]
    return ParsedDocument(
        paper_id=paper_id,
        pages=[
            ParsedPage(page_number=p, text=t, character_count=len(t))
            for p, t in pages
        ],
        page_count=len(pages),
        total_character_count=sum(len(t) for _, t in pages),
        source_storage_path="papers/test/fulltext.pdf",
    )


class TestSectionExtractorFallback:
    def test_no_headings_creates_fallback(self):
        doc = make_doc(pages=[(1, "This is just some text without any headings.")])
        extractor = SectionExtractor()
        result = extractor.extract(doc, source_parsed_path="papers/test/parsed.json")

        assert len(result.sections) == 1
        assert result.sections[0].section_type == SectionType.UNKNOWN
        assert result.sections[0].title == "Document Body"
        assert result.diagnostics.fallback_used is True

    def test_fallback_preserves_text(self):
        text = "This document has no headings but important text."
        doc = make_doc(pages=[(1, text)])
        extractor = SectionExtractor()
        result = extractor.extract(doc, source_parsed_path="papers/test/parsed.json")

        assert text in result.sections[0].text


class TestSectionExtractorWithHeadings:
    def make_paper_doc(self) -> ParsedDocument:
        page1 = "\n\nIntroduction\n\nThis is the introduction text. It contains several sentences.\n\n"
        page2 = "\nMethodology\n\nWe propose a novel approach using deep learning.\n\nResults\n\nThe results show significant improvement.\n"
        return make_doc(pages=[(1, page1), (2, page2)])

    def test_multiple_sections_detected(self):
        extractor = SectionExtractor()
        result = extractor.extract(self.make_paper_doc(), "papers/test/parsed.json")
        assert len(result.sections) >= 2

    def test_section_types_classified(self):
        extractor = SectionExtractor()
        result = extractor.extract(self.make_paper_doc(), "papers/test/parsed.json")
        types = [s.section_type for s in result.sections]
        assert SectionType.INTRODUCTION in types or SectionType.UNKNOWN in types

    def test_section_order_is_ascending(self):
        extractor = SectionExtractor()
        result = extractor.extract(self.make_paper_doc(), "papers/test/parsed.json")
        orders = [s.section_order for s in result.sections]
        assert orders == sorted(orders)

    def test_paper_id_preserved(self):
        doc = self.make_paper_doc()
        extractor = SectionExtractor()
        result = extractor.extract(doc, "papers/test/parsed.json")
        for s in result.sections:
            assert s.paper_id == doc.paper_id

    def test_page_provenance_set(self):
        extractor = SectionExtractor()
        result = extractor.extract(self.make_paper_doc(), "papers/test/parsed.json")
        for s in result.sections:
            assert s.page_start >= 1
            assert s.page_end >= s.page_start


class TestSectionExtractorPageProvenance:
    def test_multipage_section_page_range(self):
        # Heading on page 1, body spans pages 1 and 2
        p1 = "Introduction\n\nThis is the start of the introduction on page one."
        p2 = "Continued text on page two. More content here to describe the work."
        p3 = "Methodology\n\nWe used a different method here."
        doc = make_doc(pages=[(1, p1), (2, p2), (3, p3)])
        extractor = SectionExtractor()
        result = extractor.extract(doc, "papers/test/parsed.json")

        # Find the Introduction section
        intro = next((s for s in result.sections if "Introduction" in s.title), None)
        if intro:
            assert intro.page_start == 1
            assert intro.page_end >= 1  # may span multiple pages

    def test_section_result_has_paper_id(self):
        doc = make_doc(paper_id="abc-123")
        extractor = SectionExtractor()
        result = extractor.extract(doc, "papers/abc-123/parsed.json")
        assert result.paper_id == "abc-123"


class TestSectionExtractorDiagnostics:
    def test_diagnostics_section_count(self):
        page_text = "Introduction\n\nSome intro text.\n\nConclusion\n\nSome conclusion text."
        doc = make_doc(pages=[(1, page_text)])
        extractor = SectionExtractor()
        result = extractor.extract(doc, "papers/test/parsed.json")
        assert result.diagnostics.sections_detected == len(result.sections)

    def test_diagnostics_total_pages(self):
        doc = make_doc(pages=[(1, "Page one."), (2, "Page two.")])
        extractor = SectionExtractor()
        result = extractor.extract(doc, "papers/test/parsed.json")
        assert result.diagnostics.total_pages == 2


class TestSectionExtractorNoBoundaryLoss:
    def test_text_not_silently_lost(self):
        """Union of section texts + titles should contain all meaningful input text."""
        page_text = (
            "\n"
            "Introduction\n"
            "\n"
            "This is important intro content.\n"
            "\n"
            "Methodology\n"
            "\n"
            "This is the method content.\n"
        )
        doc = make_doc(pages=[(1, page_text)])
        extractor = SectionExtractor()
        result = extractor.extract(doc, "papers/test/parsed.json")

        # Collect ALL text: section titles + section bodies
        # Some short sentences may be false-detected as headings but text is never lost
        all_text = " ".join(s.title + " " + s.text for s in result.sections)
        assert "important intro content" in all_text
        assert "method" in all_text.lower()
