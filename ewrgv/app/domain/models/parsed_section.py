"""
app/domain/models/parsed_section.py
-------------------------------------
Domain models for Phase 2.6 — Section Extraction.

Design notes
------------
* ParsedSection carries the full extracted text, page provenance, heading
  metadata, and semantic classification for one section of an academic paper.
* SectionExtractionResult is the top-level container returned by SectionExtractor
  and serialised to ``sections.json`` in Supabase Storage.
* All models use Pydantic BaseModel for consistency with the rest of the project.
* PDF binary content is NEVER stored here.
* The original heading title is always preserved exactly as found.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from app.domain.enums import SectionType


class ParsedSection(BaseModel):
    """
    A single extracted section from an academic paper.

    Attributes
    ----------
    section_id:
        Unique identifier for this section (UUIDv4).
    paper_id:
        UUID of the canonical paper record.
    title:
        Original heading text exactly as found in the document.
        Example: "3.1 Experimental Setup"
    normalized_title:
        Heading with numbering prefix stripped and title-cased.
        Example: "Experimental Setup"
    section_type:
        Semantic classification from the SectionType controlled vocabulary.
    level:
        Heading hierarchy depth.  Level 1 is top-level (e.g. "3 Methodology").
        Level 2 is a subsection (e.g. "3.1 Dataset").  0 means undetermined.
    parent_section_id:
        section_id of the parent section, or None if top-level or undetectable.
    section_order:
        Zero-based index of this section in document order.
    page_start:
        1-based page number where this section begins.
    page_end:
        1-based page number where this section ends.
    text:
        Full extracted text body for this section (not including the heading).
    character_count:
        Length of ``text`` in Unicode characters.
    heading_confidence:
        Confidence score [0.0, 1.0] for the heading detection.
        1.0 = certain (e.g. matched known vocabulary + numbered pattern).
        <0.5 = low confidence fallback heading.
    """

    section_id: str = Field(default_factory=lambda: str(uuid4()))
    paper_id: str
    title: str
    normalized_title: str
    section_type: SectionType = SectionType.UNKNOWN
    level: int = 1
    parent_section_id: Optional[str] = None
    section_order: int = 0
    page_start: int = 1
    page_end: int = 1
    text: str = ""
    character_count: int = 0
    heading_confidence: float = 1.0

    @classmethod
    def build(
        cls,
        paper_id: str,
        title: str,
        normalized_title: str,
        section_type: SectionType,
        level: int,
        section_order: int,
        page_start: int,
        page_end: int,
        text: str,
        heading_confidence: float = 1.0,
        parent_section_id: Optional[str] = None,
    ) -> "ParsedSection":
        """Convenience constructor that computes character_count automatically."""
        return cls(
            paper_id=paper_id,
            title=title,
            normalized_title=normalized_title,
            section_type=section_type,
            level=level,
            parent_section_id=parent_section_id,
            section_order=section_order,
            page_start=page_start,
            page_end=page_end,
            text=text,
            character_count=len(text),
            heading_confidence=heading_confidence,
        )


class ExtractionDiagnostics(BaseModel):
    """
    Engineering diagnostics produced during section extraction.

    These are for monitoring and iterative improvement.
    They are NOT research evaluation metrics.
    """
    total_pages: int = 0
    total_input_characters: int = 0
    sections_detected: int = 0
    sections_by_type: dict[str, int] = Field(default_factory=dict)
    sections_with_text: int = 0
    low_confidence_headings: int = 0
    fallback_used: bool = False
    heading_patterns_found: list[str] = Field(default_factory=list)


class SectionExtractionResult(BaseModel):
    """
    Top-level container for section extraction output (Phase 2.6).

    This is serialised to ``sections.json`` in Supabase Storage alongside
    ``fulltext.pdf`` and ``parsed.json``.

    Attributes
    ----------
    paper_id:
        UUID of the paper.
    source_parsed_path:
        Bucket-relative path of the ``parsed.json`` used as input.
    sections:
        Ordered list of extracted sections.
    diagnostics:
        Engineering diagnostics for monitoring and improvement.
    """

    paper_id: str
    source_parsed_path: str
    sections: list[ParsedSection] = Field(default_factory=list)
    diagnostics: ExtractionDiagnostics = Field(default_factory=ExtractionDiagnostics)
