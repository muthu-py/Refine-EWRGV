"""
app/retrieval/filters.py
--------------------------
Result filters for retrieval: noise exclusion and section / paper / year filters.

Noise
-----
Reference lists, acknowledgments and publisher boilerplate (licence lines,
preprint banners, journal headers) carry no evidence for gap detection but
match queries well (they repeat title words).  They are excluded by default:

* chunks whose section_type is REFERENCES or ACKNOWLEDGMENTS, and
* chunks whose *text* is mostly bibliography or boilerplate, which catches
  reference lists and page headers the sectioner labelled OTHER.

Presets
-------
``gap_evidence`` expands to the section types where papers state limitations,
open problems and results: LIMITATIONS, FUTURE_WORK, DISCUSSION, CONCLUSION,
RESULTS.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from app.domain.models.retrieval_result import RetrievalResult

EXCLUDED_SECTION_TYPES = frozenset({"REFERENCES", "ACKNOWLEDGMENTS"})

SECTION_PRESETS: dict[str, frozenset[str]] = {
    "gap_evidence": frozenset({"LIMITATIONS", "FUTURE_WORK", "DISCUSSION", "CONCLUSION", "RESULTS"}),
}

# Citation markers typical of reference entries.
_RE_CITATION_MARK = re.compile(
    # "Smith et al. (2020)" is one narrative citation, not two markers
    r"\bet al\b\.?(?:,?\s*\(?(?:19|20)\d{2}[a-z]?\)?)?|\((?:19|20)\d{2}[a-z]?\)|\b(?:19|20)\d{2}[a-z]?;|\bdoi\b|https?://|\bvol\.|\bpp\.|"
    r"\bproceedings of\b|\barxiv\b|\bjournal of\b|\bin:\s|\bieee trans|\b\d+\(\d+\):\d+",
    re.IGNORECASE,
)
# Numbered reference entries: "[12] ", "12. Smith, J", "\t37. Boureau"
_RE_REF_ENTRY = re.compile(r"(?:^|\s)(?:\[\d{1,3}\]|\d{1,3}\.)\s+[A-Z][a-zA-Z'\-]+,?\s+[A-Z]{1,2}\b")
# Publisher / repository boilerplate
_RE_BOILERPLATE = re.compile(
    r"\bcc[- ]by\b|\blicen[cs]e\b|\bcopyright\b|\bpreprint\b|\ball rights reserved\b|"
    r"\bpeer review\b|\bissn\b|\bvolume\s+\d+\b|\bissue\s+\d+\b|\bcorresponding (?:author|email)\b|"
    r"\breceived\b.{0,20}\baccepted\b|\barticle history\b|\bpublished online\b",
    re.IGNORECASE,
)


def citation_density(text: str) -> float:
    """Citation markers + reference entries per 100 words."""
    words = max(1, len(text.split()))
    marks = len(_RE_CITATION_MARK.findall(text)) + 2 * len(_RE_REF_ENTRY.findall(text))
    return 100.0 * marks / words


def is_noise_text(text: str) -> bool:
    """True when a chunk's text is mostly bibliography or publisher boilerplate."""
    words = len(text.split())
    if words == 0:
        return True
    if citation_density(text) >= 6.0:
        return True
    # Boilerplate must dominate: a real conclusion with a licence banner
    # attached (medRxiv, PLOS footers) must survive.
    boilerplate = len(_RE_BOILERPLATE.findall(text))
    return boilerplate >= 3 and boilerplate * 12 >= words


def is_noise(section_type: str, text: str) -> bool:
    return (section_type or "").upper() in EXCLUDED_SECTION_TYPES or is_noise_text(text)


def expand_section_types(section_types: Optional[Iterable[str]]) -> Optional[frozenset[str]]:
    """Upper-case section types, expanding preset names; None/empty means no filter."""
    if not section_types:
        return None
    out: set[str] = set()
    for name in section_types:
        key = (name or "").strip()
        if not key:
            continue
        preset = SECTION_PRESETS.get(key.lower())
        out |= preset if preset else {key.upper()}
    return frozenset(out) or None


@dataclass(frozen=True)
class RetrievalFilters:
    """
    Filters applied to ranked results.

    section_types : keep only these section types (presets already expanded)
    paper_ids     : keep only these papers
    year_from/to  : inclusive publication-year bounds (results without a year
                    are dropped when a bound is set)
    exclude_noise : drop references / acknowledgments / boilerplate (default on)
    """

    section_types: Optional[frozenset[str]] = None
    paper_ids: Optional[frozenset[str]] = None
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    exclude_noise: bool = True

    @classmethod
    def build(
        cls,
        section_types: Optional[Iterable[str]] = None,
        paper_ids: Optional[Iterable[str]] = None,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        exclude_noise: bool = True,
    ) -> "RetrievalFilters":
        return cls(
            section_types=expand_section_types(section_types),
            paper_ids=frozenset(str(p) for p in paper_ids) if paper_ids else None,
            year_from=year_from,
            year_to=year_to,
            exclude_noise=exclude_noise,
        )

    @property
    def active(self) -> bool:
        return bool(
            self.exclude_noise or self.section_types or self.paper_ids
            or self.year_from is not None or self.year_to is not None
        )

    def keep(self, r: RetrievalResult) -> bool:
        section_type = (r.section_type or "").upper()
        if self.exclude_noise and is_noise(section_type, r.text):
            return False
        if self.section_types is not None and section_type not in self.section_types:
            return False
        if self.paper_ids is not None and r.paper_id not in self.paper_ids:
            return False
        if self.year_from is not None or self.year_to is not None:
            year = (r.metadata or {}).get("publication_year")
            if not isinstance(year, int):
                return False
            if self.year_from is not None and year < self.year_from:
                return False
            if self.year_to is not None and year > self.year_to:
                return False
        return True

    def apply(self, results: list[RetrievalResult]) -> list[RetrievalResult]:
        return [r for r in results if self.keep(r)] if self.active else list(results)


NO_FILTERS = RetrievalFilters(exclude_noise=False)
