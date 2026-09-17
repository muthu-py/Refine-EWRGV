"""
app/domain/models/parsed_document.py
-------------------------------------
Domain models for a parsed academic paper document.

These models represent the output of Phase 2.5 — Document Parsing.
They carry page-level text and basic provenance so that later phases
(section extraction, chunking, retrieval) can trace evidence back to
a specific page in the original PDF.

Design notes
------------
* ParsedDocument is intentionally lightweight: plain text per page only.
* PDF binary content is NEVER stored here; it lives in Supabase Storage.
* The canonical Paper (bibliographic metadata) remains the authority for
  author, title, DOI, etc.  ParsedDocument.metadata holds only what the
  PDF itself exposes (creator, producer, pdf title, etc.).
* Pydantic BaseModel is used for consistency with all other domain models.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class ParsedPage(BaseModel):
    """
    Text content extracted from a single PDF page.

    Attributes
    ----------
    page_number:
        1-based page number (page 1 is the first page of the document).
    text:
        Raw extracted text for this page.  May be empty for blank or
        image-only pages.
    character_count:
        Length of `text` in Unicode characters (not bytes).
    """

    page_number: int
    text: str
    character_count: int

    @classmethod
    def from_text(cls, page_number: int, text: str) -> "ParsedPage":
        """Convenience constructor that computes character_count automatically."""
        return cls(page_number=page_number, text=text, character_count=len(text))


class ParsedDocument(BaseModel):
    """
    Structured representation of a parsed PDF document.

    This is the primary output of Phase 2.5.  It is serialised to
    ``parsed.json`` in Supabase Storage alongside the original PDF.

    Attributes
    ----------
    paper_id:
        UUID of the canonical paper record this document belongs to.
    source_storage_path:
        The Supabase Storage path of the original PDF that was parsed.
        Format: ``papers/{paper_id}/fulltext.pdf`` (bucket-relative).
    page_count:
        Total number of pages in the PDF.
    total_character_count:
        Sum of character counts across all pages.  Useful for detecting
        scanned-only PDFs with no text layer.
    pages_with_text:
        Number of pages that contain at least one non-whitespace character.
    metadata:
        PDF-level metadata as reported by the PDF library (title, author,
        creator, producer, creation_date, etc.).  NOT authoritative
        bibliographic metadata — that lives in the Paper record.
    pages:
        Ordered list of ParsedPage objects (one per PDF page).
    """

    paper_id: str
    source_storage_path: str
    page_count: int
    total_character_count: int = 0
    pages_with_text: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    pages: list[ParsedPage] = Field(default_factory=list)

    @classmethod
    def build(
        cls,
        paper_id: str,
        source_storage_path: str,
        pages: list[ParsedPage],
        metadata: Optional[dict[str, Any]] = None,
    ) -> "ParsedDocument":
        """
        Construct a ParsedDocument from a list of ParsedPage objects,
        automatically computing aggregate statistics.
        """
        total_chars = sum(p.character_count for p in pages)
        pages_with_text = sum(
            1 for p in pages if p.text.strip()
        )
        return cls(
            paper_id=paper_id,
            source_storage_path=source_storage_path,
            page_count=len(pages),
            total_character_count=total_chars,
            pages_with_text=pages_with_text,
            metadata=metadata or {},
            pages=pages,
        )
