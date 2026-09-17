"""
app/domain/models/paper_chunk.py
----------------------------------
Domain models for Phase 2.7 — Chunking.

Design notes
------------
* PaperChunk is the atomic retrieval unit. It carries the actual paper text
  plus full provenance back to: section → page → original PDF.
* ChunkingResult is the top-level container serialised to ``chunks.json``.
* All models use Pydantic BaseModel for consistency.
* PDF binary is NEVER stored here.
* No LLM-generated text, no paraphrasing — original paper text only.

Provenance chain:
    chunk_id → paper_id → section_id → page_start/page_end → PDF
"""

from __future__ import annotations

from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from app.domain.enums import SectionType


class PaperChunk(BaseModel):
    """
    A single retrieval-ready text chunk from an academic paper.

    Attributes
    ----------
    chunk_id:
        Unique identifier (UUIDv4).
    paper_id:
        UUID of the canonical paper record.
    section_id:
        UUID of the ParsedSection this chunk belongs to.
    section_title:
        Original heading title of the source section.
    section_type:
        Semantic type of the source section.
    section_level:
        Hierarchy depth of the source section.
    chunk_index:
        Zero-based position of this chunk within its section.
    text:
        Actual extracted paper text. Never paraphrased or rewritten.
    page_start:
        1-based page number where this chunk's text begins.
    page_end:
        1-based page number where this chunk's text ends.
    token_count:
        Approximate token count (character-based estimate: chars // 4).
    character_count:
        Length of text in Unicode characters.
    char_start:
        Character offset within the section text where this chunk starts.
    char_end:
        Character offset within the section text where this chunk ends.
    """

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    paper_id: str
    section_id: str
    section_title: str
    section_type: SectionType = SectionType.UNKNOWN
    section_level: int = 1
    chunk_index: int = 0
    text: str
    page_start: int = 1
    page_end: int = 1
    token_count: int = 0
    character_count: int = 0
    char_start: int = 0
    char_end: int = 0

    @classmethod
    def build(
        cls,
        paper_id: str,
        section_id: str,
        section_title: str,
        section_type: SectionType,
        section_level: int,
        chunk_index: int,
        text: str,
        page_start: int,
        page_end: int,
        char_start: int = 0,
        char_end: int = 0,
    ) -> "PaperChunk":
        """Convenience constructor that computes token_count and character_count."""
        char_count = len(text)
        # Lightweight token approximation: ~4 characters per token (GPT-style).
        # This avoids a tokenizer dependency. Documented as approximate.
        token_approx = max(1, char_count // 4)
        return cls(
            paper_id=paper_id,
            section_id=section_id,
            section_title=section_title,
            section_type=section_type,
            section_level=section_level,
            chunk_index=chunk_index,
            text=text,
            page_start=page_start,
            page_end=page_end,
            token_count=token_approx,
            character_count=char_count,
            char_start=char_start,
            char_end=char_end,
        )


class ChunkingDiagnostics(BaseModel):
    """Engineering diagnostics produced during chunking."""
    total_sections: int = 0
    total_chunks: int = 0
    chunks_by_section_type: dict[str, int] = Field(default_factory=dict)
    oversized_chunks: int = 0
    empty_chunks_skipped: int = 0
    hard_splits_used: int = 0
    total_characters: int = 0
    avg_chunk_tokens: float = 0.0


class ChunkingResult(BaseModel):
    """
    Top-level container for Phase 2.7 chunking output.

    Serialised to ``chunks.json`` in Supabase Storage.

    Storage path: papers/{paper_id}/chunks.json
    """
    paper_id: str
    source_sections_path: str
    chunks: list[PaperChunk] = Field(default_factory=list)
    diagnostics: ChunkingDiagnostics = Field(default_factory=ChunkingDiagnostics)
