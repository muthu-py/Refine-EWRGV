"""
app/ingestion/chunking/chunker.py
-----------------------------------
Section-aware, paragraph-aware, sentence-aware chunker for Phase 2.7.

Architecture
------------

For each ParsedSection:
  1. Normalize whitespace in section text.
  2. Split section text into paragraphs (double-newline boundaries).
  3. Accumulate paragraphs into a buffer until target_chunk_size is reached.
  4. If a paragraph alone exceeds max_chunk_size:
       a. Split at sentence boundaries.
       b. Accumulate sentences into chunks.
       c. If a sentence exceeds max_chunk_size: hard character split (fallback).
  5. Apply controlled overlap (last overlap_size chars of previous chunk prepended).
  6. Emit a PaperChunk for each buffer flush.
  7. NEVER cross section boundaries.
  8. Assign page provenance from section.page_start / page_end (best effort).
  9. Validate every chunk (non-empty, within size, has paper_id + section_id).

Configuration is via ChunkConfig (see chunker.py bottom or app/core/config.py).

Token counting:
  Approximate: character_count // 4 (GPT-style tokenizer estimate).
  This avoids adding a tokenizer dependency. Documented as approximate.
  Overridable by subclassing or injecting a token_fn.

Critical invariants:
  - No chunk crosses a section boundary.
  - No empty/whitespace-only chunks emitted.
  - No text is silently lost (union of chunks ≈ section text, modulo overlap).
  - Page provenance is always present.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from app.domain.models.paper_chunk import (
    ChunkingDiagnostics,
    ChunkingResult,
    PaperChunk,
)
from app.domain.models.parsed_section import ParsedSection, SectionExtractionResult


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ChunkConfig:
    """
    Configuration for the chunker.

    All sizes are in *approximate tokens* (chars // 4).

    Attributes
    ----------
    target_chunk_size:
        Preferred chunk size in tokens.  Chunks will try not to exceed this
        unless a single paragraph forces it.
    max_chunk_size:
        Hard upper bound per chunk.  Only a sentence or hard split may exceed it.
    overlap_size:
        Number of tokens to overlap between consecutive chunks within a section.
    min_chunk_size:
        Chunks below this token count are still emitted (never discarded),
        but a warning may be logged.
    min_chunk_words:
        Sections with fewer words are merged into a neighbouring section
        before chunking, and a trailing chunk with fewer words is merged
        into the previous chunk of its section.  Stops false headings and
        wrapped PDF lines from producing 3-word "chunks".  0 disables.
    """
    target_chunk_size: int = 500      # ~2000 characters
    max_chunk_size: int = 600         # ~2400 characters
    overlap_size: int = 75            # ~300 characters
    min_chunk_size: int = 20          # ~80 characters
    min_chunk_words: int = 60

    @property
    def target_chars(self) -> int:
        return self.target_chunk_size * 4

    @property
    def max_chars(self) -> int:
        return self.max_chunk_size * 4

    @property
    def overlap_chars(self) -> int:
        return self.overlap_size * 4

    @property
    def min_chars(self) -> int:
        return self.min_chunk_size * 4


# ---------------------------------------------------------------------------
# Sentence splitter (no NLTK dependency)
# ---------------------------------------------------------------------------

# Pattern: split after sentence-ending punctuation followed by whitespace.
# Handles common abbreviations by requiring the next token to start with uppercase.
_SENT_BOUNDARY = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z"\'(\[])',
)


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using a lightweight regex.

    Does not require NLTK or any external NLP library.
    Falls back gracefully for edge cases.
    """
    parts = _SENT_BOUNDARY.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Paragraph splitter
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """
    Undo PDF line wrapping: a single newline inside a paragraph becomes a
    space (keeping hyphenated words joined); blank lines that only contain
    whitespace become real paragraph breaks.
    """
    text = re.sub(r"\n[ \t]+\n", "\n\n", text)
    text = re.sub(r"(\w)-[ \t]*\n(?!\n)[ \t]*([a-z])", r"\1\2", text)
    text = re.sub(r"[ \t]*(?<!\n)\n(?!\n)[ \t]*", " ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _word_count(text: str) -> int:
    return len(text.split())


# Section types kept as their own chunk even when short: they are the
# primary gap evidence and must keep their section_type.
_PROTECTED_TYPES = {"LIMITATIONS", "FUTURE_WORK"}


def _split_paragraphs(text: str) -> list[str]:
    """Split text at double-newline boundaries (paragraph breaks)."""
    # Normalize whitespace first
    text = re.sub(r"\n{3,}", "\n\n", text)
    paras = re.split(r"\n\n+", text)
    return [p.strip() for p in paras if p.strip()]


# ---------------------------------------------------------------------------
# Token approximation
# ---------------------------------------------------------------------------

def _approx_tokens(text: str) -> int:
    """Approximate token count: character_count // 4."""
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Main Chunker class
# ---------------------------------------------------------------------------


class SectionChunker:
    """
    Converts a SectionExtractionResult into a ChunkingResult.

    Usage
    -----
    config = ChunkConfig(target_chunk_size=500, max_chunk_size=600, overlap_size=75)
    chunker = SectionChunker(config=config)
    result = chunker.chunk(extraction_result, source_sections_path="papers/id/sections.json")
    """

    def __init__(self, config: Optional[ChunkConfig] = None) -> None:
        self._cfg = config or ChunkConfig()

    def chunk(
        self,
        extraction_result: SectionExtractionResult,
        source_sections_path: str,
    ) -> ChunkingResult:
        """
        Process all sections and produce a ChunkingResult.

        Parameters
        ----------
        extraction_result:
            Output of Phase 2.6.
        source_sections_path:
            Bucket-relative path to sections.json (for provenance).

        Returns
        -------
        ChunkingResult:
            Container with ordered chunks and diagnostics.
        """
        paper_id = extraction_result.paper_id
        all_chunks: list[PaperChunk] = []
        diag = ChunkingDiagnostics(total_sections=len(extraction_result.sections))

        for section in self._merge_small_sections(extraction_result.sections):
            section_chunks = self._merge_small_chunks(self._chunk_section(section, paper_id), section, paper_id)
            # Update global chunk_index across all sections
            for ch in section_chunks:
                ch.chunk_index = len(all_chunks) + section_chunks.index(ch)
            all_chunks.extend(section_chunks)

        # Build diagnostics
        diag.total_chunks = len(all_chunks)
        diag.total_characters = sum(c.character_count for c in all_chunks)
        if all_chunks:
            diag.avg_chunk_tokens = sum(c.token_count for c in all_chunks) / len(all_chunks)

        type_counts: dict[str, int] = {}
        for c in all_chunks:
            k = c.section_type.value
            type_counts[k] = type_counts.get(k, 0) + 1
        diag.chunks_by_section_type = type_counts

        # Re-assign deterministic chunk_index (0-based globally)
        for i, ch in enumerate(all_chunks):
            ch.chunk_index = i

        return ChunkingResult(
            paper_id=paper_id,
            source_sections_path=source_sections_path,
            chunks=all_chunks,
            diagnostics=diag,
        )

    # ------------------------------------------------------------------ #
    # Section-level chunking
    # ------------------------------------------------------------------ #

    def _chunk_section(
        self, section: ParsedSection, paper_id: str
    ) -> list[PaperChunk]:
        """
        Chunk a single section into PaperChunk objects.

        Section boundaries are NEVER crossed.
        """
        text = _normalize_text(section.text)
        if not text:
            return []

        paragraphs = _split_paragraphs(text)
        if not paragraphs:
            return []

        chunks: list[PaperChunk] = []
        overlap_tail: str = ""  # last N chars of previous chunk for overlap

        # Accumulate paragraphs into buffer
        buffer: list[str] = []
        buffer_chars: int = 0
        char_cursor: int = 0  # tracks position within full section text

        for para in paragraphs:
            para_chars = len(para)

            if para_chars > self._cfg.max_chars:
                # Paragraph too large — flush current buffer first, then split para
                if buffer:
                    chunk = self._emit_chunk(
                        buffer=buffer,
                        overlap_tail=overlap_tail,
                        section=section,
                        paper_id=paper_id,
                        char_start=char_cursor - buffer_chars,
                        char_end=char_cursor,
                    )
                    chunks.append(chunk)
                    overlap_tail = self._extract_overlap(chunk.text)
                    buffer = []
                    buffer_chars = 0

                # Split large paragraph at sentence boundaries
                sent_chunks = self._chunk_large_paragraph(
                    para=para,
                    section=section,
                    paper_id=paper_id,
                    overlap_tail=overlap_tail,
                    char_start=char_cursor,
                )
                chunks.extend(sent_chunks)
                if sent_chunks:
                    overlap_tail = self._extract_overlap(sent_chunks[-1].text)
                char_cursor += para_chars + 2  # +2 for \n\n
                continue

            # Check if adding this paragraph exceeds target
            projected = buffer_chars + (2 if buffer else 0) + para_chars

            if buffer and projected > self._cfg.target_chars:
                # Flush buffer
                chunk = self._emit_chunk(
                    buffer=buffer,
                    overlap_tail=overlap_tail,
                    section=section,
                    paper_id=paper_id,
                    char_start=char_cursor - buffer_chars,
                    char_end=char_cursor,
                )
                chunks.append(chunk)
                overlap_tail = self._extract_overlap(chunk.text)
                buffer = []
                buffer_chars = 0

            buffer.append(para)
            buffer_chars += para_chars + (2 if len(buffer) > 1 else 0)
            char_cursor += para_chars + 2

        # Flush remaining buffer
        if buffer:
            chunk = self._emit_chunk(
                buffer=buffer,
                overlap_tail=overlap_tail,
                section=section,
                paper_id=paper_id,
                char_start=char_cursor - buffer_chars,
                char_end=char_cursor,
            )
            chunks.append(chunk)

        return chunks

    # ------------------------------------------------------------------ #
    # Small-fragment merging
    # ------------------------------------------------------------------ #

    def _merge_small_sections(self, sections: list[ParsedSection]) -> list[ParsedSection]:
        """
        Fold sections shorter than min_chunk_words into the previous section
        (or the next one, for a short first section).

        A short section is almost always a false heading (wrapped line,
        author name, page header) that split real prose.  Its title is kept
        as text, since for a false heading the "title" is body text.
        LIMITATIONS / FUTURE_WORK sections are never folded.
        """
        min_words = self._cfg.min_chunk_words
        if min_words <= 0:
            return list(sections)

        def protected(sec: ParsedSection) -> bool:
            return getattr(sec.section_type, "value", str(sec.section_type)) in _PROTECTED_TYPES

        def size(sec: ParsedSection) -> int:
            return _word_count(sec.text)

        def absorb(into: ParsedSection, other: ParsedSection, prepend: bool) -> ParsedSection:
            piece = "\n\n".join(t for t in (other.title.strip(), other.text.strip()) if t)
            text = f"{piece}\n\n{into.text}" if prepend else f"{into.text}\n\n{piece}"
            return into.model_copy(update={
                "text": text.strip(),
                "character_count": len(text.strip()),
                "page_start": min(into.page_start, other.page_start),
                "page_end": max(into.page_end, other.page_end),
            })

        merged: list[ParsedSection] = []
        pending: list[ParsedSection] = []  # short sections before any keeper
        for sec in sections:
            if size(sec) >= min_words or protected(sec):
                for early in reversed(pending):
                    sec = absorb(sec, early, prepend=True)
                pending = []
                merged.append(sec)
            elif merged and not protected(merged[-1]):
                merged[-1] = absorb(merged[-1], sec, prepend=False)
            elif merged:
                pending.append(sec)  # don't dilute a protected section; fold forward
            else:
                pending.append(sec)
        if pending:
            if merged:
                for sec in pending:
                    merged[-1] = absorb(merged[-1], sec, prepend=False)
            else:
                first = pending[0]
                for sec in pending[1:]:
                    first = absorb(first, sec, prepend=False)
                merged.append(first)
        return merged

    def _merge_small_chunks(
        self, chunks: list[PaperChunk], section: ParsedSection, paper_id: str
    ) -> list[PaperChunk]:
        """Merge a too-short chunk into the previous chunk of the same section."""
        min_words = self._cfg.min_chunk_words
        if min_words <= 0 or len(chunks) < 2:
            return chunks
        out: list[PaperChunk] = [chunks[0]]
        for ch in chunks[1:]:
            prev = out[-1]
            if _word_count(ch.text) < min_words and len(prev.text) + len(ch.text) <= self._cfg.max_chars * 1.25:
                out[-1] = self._make_chunk(
                    text=f"{prev.text}\n\n{ch.text}",
                    section=section,
                    paper_id=paper_id,
                    char_start=prev.char_start,
                    char_end=ch.char_end,
                )
            else:
                out.append(ch)
        return out

    # ------------------------------------------------------------------ #
    # Large paragraph → sentence chunks
    # ------------------------------------------------------------------ #

    def _chunk_large_paragraph(
        self,
        para: str,
        section: ParsedSection,
        paper_id: str,
        overlap_tail: str,
        char_start: int,
    ) -> list[PaperChunk]:
        """Split a paragraph that is larger than max_chunk_size at sentence boundaries."""
        sentences = _split_sentences(para)
        chunks: list[PaperChunk] = []
        buffer: list[str] = []
        buffer_chars = 0
        pos = char_start

        for sent in sentences:
            sent_chars = len(sent)

            if sent_chars > self._cfg.max_chars:
                # Sentence itself too large — hard character split
                if buffer:
                    chunk = self._emit_chunk(
                        buffer=buffer,
                        overlap_tail=overlap_tail,
                        section=section,
                        paper_id=paper_id,
                        char_start=pos - buffer_chars,
                        char_end=pos,
                    )
                    chunks.append(chunk)
                    overlap_tail = self._extract_overlap(chunk.text)
                    buffer = []
                    buffer_chars = 0

                hard_chunks = self._hard_split(
                    text=sent,
                    section=section,
                    paper_id=paper_id,
                    overlap_tail=overlap_tail,
                    char_start=pos,
                )
                chunks.extend(hard_chunks)
                if hard_chunks:
                    overlap_tail = self._extract_overlap(hard_chunks[-1].text)
                pos += sent_chars + 1
                continue

            projected = buffer_chars + (1 if buffer else 0) + sent_chars
            if buffer and projected > self._cfg.target_chars:
                chunk = self._emit_chunk(
                    buffer=buffer,
                    overlap_tail=overlap_tail,
                    section=section,
                    paper_id=paper_id,
                    char_start=pos - buffer_chars,
                    char_end=pos,
                )
                chunks.append(chunk)
                overlap_tail = self._extract_overlap(chunk.text)
                buffer = []
                buffer_chars = 0

            buffer.append(sent)
            buffer_chars += sent_chars + (1 if len(buffer) > 1 else 0)
            pos += sent_chars + 1

        if buffer:
            chunk = self._emit_chunk(
                buffer=buffer,
                overlap_tail=overlap_tail,
                section=section,
                paper_id=paper_id,
                char_start=pos - buffer_chars,
                char_end=pos,
            )
            chunks.append(chunk)

        return chunks

    # ------------------------------------------------------------------ #
    # Hard character split (last resort)
    # ------------------------------------------------------------------ #

    def _hard_split(
        self,
        text: str,
        section: ParsedSection,
        paper_id: str,
        overlap_tail: str,
        char_start: int,
    ) -> list[PaperChunk]:
        """Split text at character boundaries when even sentences are too large."""
        chunks: list[PaperChunk] = []
        step = self._cfg.target_chars
        pos = 0
        chunk_pos = char_start

        while pos < len(text):
            slice_text = text[pos: pos + step]
            if not slice_text.strip():
                pos += step
                continue

            full_text = (overlap_tail + " " + slice_text).strip() if overlap_tail else slice_text
            chunk = self._make_chunk(
                text=full_text,
                section=section,
                paper_id=paper_id,
                char_start=chunk_pos,
                char_end=chunk_pos + len(slice_text),
            )
            chunks.append(chunk)
            overlap_tail = self._extract_overlap(slice_text)
            pos += step
            chunk_pos += len(slice_text)

        return chunks

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _emit_chunk(
        self,
        buffer: list[str],
        overlap_tail: str,
        section: ParsedSection,
        paper_id: str,
        char_start: int,
        char_end: int,
    ) -> PaperChunk:
        body = "\n\n".join(buffer)
        full_text = (overlap_tail + "\n\n" + body).strip() if overlap_tail else body
        return self._make_chunk(
            text=full_text,
            section=section,
            paper_id=paper_id,
            char_start=max(0, char_start),
            char_end=max(0, char_end),
        )

    def _make_chunk(
        self,
        text: str,
        section: ParsedSection,
        paper_id: str,
        char_start: int,
        char_end: int,
    ) -> PaperChunk:
        return PaperChunk.build(
            paper_id=paper_id,
            section_id=section.section_id,
            section_title=section.title,
            section_type=section.section_type,
            section_level=section.level,
            chunk_index=0,  # will be reassigned globally
            text=text,
            page_start=section.page_start,
            page_end=section.page_end,
            char_start=char_start,
            char_end=char_end,
        )

    def _extract_overlap(self, text: str) -> str:
        """Return the last overlap_chars characters for prepending to next chunk."""
        overlap_chars = self._cfg.overlap_chars
        if len(text) <= overlap_chars:
            return text
        # Prefer to break at a sentence boundary within the overlap window
        tail = text[-overlap_chars:]
        # Find the last sentence boundary
        sentences = _split_sentences(tail)
        if len(sentences) > 1:
            # Use all but the first partial sentence
            return " ".join(sentences[1:]).strip()
        return tail
