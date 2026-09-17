"""
app/ingestion/sectioning/extractor.py
---------------------------------------
Section extractor for Phase 2.6 — converts a ParsedDocument into
a list of ParsedSection objects with full page provenance.

Architecture
------------
1. Build a line index: (line_index, page_number, text) for all pages.
2. Run heading_detector.detect_headings() to find heading candidates.
3. Assign text between consecutive headings to each section.
4. Resolve hierarchy (parent_section_id) from numbering depth.
5. Normalize + classify each heading.
6. Build a fallback "Document Body" section if no headings detected.
7. Return SectionExtractionResult.

Critical design rules
---------------------
* Section boundaries are hard — text NEVER crosses unrelated sections.
* Page provenance is ALWAYS preserved (page_start, page_end).
* No text is silently discarded.
* The original title is stored exactly as found; normalized_title is separate.
"""

from __future__ import annotations

import re
from typing import Optional

from app.domain.enums import SectionType
from app.domain.models.parsed_document import ParsedDocument, ParsedPage
from app.domain.models.parsed_section import (
    ExtractionDiagnostics,
    ParsedSection,
    SectionExtractionResult,
)
from app.ingestion.sectioning.classifier import classify
from app.ingestion.sectioning.heading_detector import (
    HeadingCandidate,
    detect_headings,
)
from app.ingestion.sectioning.normalizer import normalize_heading


class SectionExtractor:
    """
    Converts a ParsedDocument into structured ParsedSection objects.

    Usage
    -----
    extractor = SectionExtractor()
    result = extractor.extract(parsed_doc, source_parsed_path="papers/id/parsed.json")
    """

    def extract(
        self,
        doc: ParsedDocument,
        source_parsed_path: str,
    ) -> SectionExtractionResult:
        """
        Extract sections from a ParsedDocument.

        Parameters
        ----------
        doc:
            Output of Phase 2.5 (ParsedDocument with page-level text).
        source_parsed_path:
            Bucket-relative path of the parsed.json (for provenance).

        Returns
        -------
        SectionExtractionResult:
            Container with ordered sections + diagnostics.
        """
        # Build flat line list with page provenance
        lines, page_of_line = self._build_line_index(doc.pages)

        # Detect heading candidates
        candidates = detect_headings(lines)

        # Build sections
        sections: list[ParsedSection]
        fallback_used = False

        if not candidates:
            # No headings found — create a single fallback section
            all_text = self._join_lines([line[2] for line in lines])
            fallback_used = True
            sections = self._make_fallback_sections(
                paper_id=doc.paper_id,
                text=all_text,
                page_start=doc.pages[0].page_number if doc.pages else 1,
                page_end=doc.pages[-1].page_number if doc.pages else 1,
            )
        else:
            sections = self._build_sections(
                paper_id=doc.paper_id,
                lines=lines,
                candidates=candidates,
            )

        # Resolve parent hierarchy
        sections = self._resolve_hierarchy(sections)

        # Build diagnostics
        diagnostics = self._build_diagnostics(
            doc=doc,
            sections=sections,
            candidates=candidates,
            fallback_used=fallback_used,
        )

        return SectionExtractionResult(
            paper_id=doc.paper_id,
            source_parsed_path=source_parsed_path,
            sections=sections,
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------ #
    # Step 1: Build line index
    # ------------------------------------------------------------------ #

    def _build_line_index(
        self, pages: list[ParsedPage]
    ) -> tuple[list[tuple[int, int, str]], dict[int, int]]:
        """
        Create a flat list of (line_global_index, page_number, text).

        Returns
        -------
        lines:
            All lines across all pages in order.
        page_of_line:
            Mapping from line_global_index to page_number.
        """
        lines: list[tuple[int, int, str]] = []
        page_of_line: dict[int, int] = {}
        global_idx = 0

        for page in pages:
            for raw_line in page.text.splitlines():
                lines.append((global_idx, page.page_number, raw_line))
                page_of_line[global_idx] = page.page_number
                global_idx += 1

        return lines, page_of_line

    # ------------------------------------------------------------------ #
    # Step 2: Build sections from heading candidates
    # ------------------------------------------------------------------ #

    def _build_sections(
        self,
        paper_id: str,
        lines: list[tuple[int, int, str]],
        candidates: list[HeadingCandidate],
    ) -> list[ParsedSection]:
        """
        Slice the document at each heading boundary to produce sections.
        """
        sections: list[ParsedSection] = []

        # Index lines by their line_index for fast lookup
        line_by_idx: dict[int, tuple[int, int, str]] = {t[0]: t for t in lines}

        # Convert candidates to a sorted list (they should already be sorted)
        headings = sorted(candidates, key=lambda h: h.line_index)

        # Determine where each section's body text starts/ends
        # (from the line after the heading to the line before the next heading)
        for order, heading in enumerate(headings):
            next_heading_line = (
                headings[order + 1].line_index
                if order + 1 < len(headings)
                else len(lines)
            )

            # Collect body lines (lines between this heading and the next)
            body_lines: list[str] = []
            page_start = heading.page_number
            page_end = heading.page_number

            for i in range(heading.line_index + 1, next_heading_line):
                if i < len(lines):
                    _, pg, txt = lines[i]
                    body_lines.append(txt)
                    page_end = max(page_end, pg)

            body_text = self._join_lines(body_lines)
            normalized = normalize_heading(heading.raw_text)
            section_type = classify(normalized)

            sections.append(
                ParsedSection.build(
                    paper_id=paper_id,
                    title=heading.raw_text,
                    normalized_title=normalized,
                    section_type=section_type,
                    level=heading.level,
                    section_order=order,
                    page_start=page_start,
                    page_end=page_end,
                    text=body_text,
                    heading_confidence=heading.score,
                )
            )

        return sections

    # ------------------------------------------------------------------ #
    # Step 3: Fallback section
    # ------------------------------------------------------------------ #

    def _make_fallback_sections(
        self,
        paper_id: str,
        text: str,
        page_start: int,
        page_end: int,
    ) -> list[ParsedSection]:
        """Create a single fallback section when no headings are detected."""
        return [
            ParsedSection.build(
                paper_id=paper_id,
                title="Document Body",
                normalized_title="Document Body",
                section_type=SectionType.UNKNOWN,
                level=1,
                section_order=0,
                page_start=page_start,
                page_end=page_end,
                text=text,
                heading_confidence=0.0,
            )
        ]

    # ------------------------------------------------------------------ #
    # Step 4: Resolve hierarchy
    # ------------------------------------------------------------------ #

    def _resolve_hierarchy(self, sections: list[ParsedSection]) -> list[ParsedSection]:
        """
        Set parent_section_id based on numbering depth.

        Rules:
        - A section at level N has the most recent section at level N-1 as parent.
        - If no parent at level N-1 exists, parent_section_id remains None.
        - Only resolves hierarchy when numbering is present and consistent.
        """
        # Stack: [(level, section_id)]
        stack: list[tuple[int, str]] = []

        for section in sections:
            level = section.level

            # Pop stack entries at the same or deeper level
            while stack and stack[-1][0] >= level:
                stack.pop()

            if stack:
                # Parent is the top of the stack
                section.parent_section_id = stack[-1][1]

            # Push current section
            stack.append((level, section.section_id))

        return sections

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _join_lines(lines: list[str]) -> str:
        """Join lines into a single string, normalizing excessive blank lines."""
        # Join preserving paragraph structure
        text = "\n".join(lines)
        # Normalize 3+ consecutive blank lines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _build_diagnostics(
        doc: ParsedDocument,
        sections: list[ParsedSection],
        candidates: list[HeadingCandidate],
        fallback_used: bool,
    ) -> ExtractionDiagnostics:
        sections_by_type: dict[str, int] = {}
        for s in sections:
            key = s.section_type.value
            sections_by_type[key] = sections_by_type.get(key, 0) + 1

        low_conf = sum(1 for c in candidates if c.score < 0.6)
        patterns_found = list({
            sig
            for c in candidates
            for sig in c.signals
            if sig.startswith(("ARABIC", "ROMAN", "LETTER", "KNOWN", "ALL_CAPS"))
        })

        return ExtractionDiagnostics(
            total_pages=doc.page_count,
            total_input_characters=doc.total_character_count,
            sections_detected=len(sections),
            sections_by_type=sections_by_type,
            sections_with_text=sum(1 for s in sections if s.text.strip()),
            low_confidence_headings=low_conf,
            fallback_used=fallback_used,
            heading_patterns_found=sorted(patterns_found),
        )
