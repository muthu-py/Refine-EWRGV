"""
app/evidence/strength.py
--------------------------
Evidence strength: how much one classified item weighs, independent of its type.

    strength = relevance x recency x source

    recency  1.0  newer than the gap's newest stating paper (fresh work counts fully)
             0.8  same year or older
             0.9  year unknown
    source   1.0  run corpus (full text, already screened for relevance)
             0.9  external title + abstract
"""

from __future__ import annotations

from typing import Optional

from app.domain.models.evidence import Evidence


def strength_of(relevance: float, year: Optional[int], reference_year: Optional[int], source: str) -> float:
    if year is None or reference_year is None:
        recency = 0.9
    else:
        recency = 1.0 if year > reference_year else 0.8
    src = 1.0 if source == "corpus" else 0.9
    return round(max(0.0, min(1.0, relevance)) * recency * src, 4)


def assess_strength(evidence: Evidence) -> float:
    """Strength of an Evidence item (uses metadata year / reference_year / source)."""
    meta = evidence.metadata or {}
    return strength_of(evidence.relevance, meta.get("year"), meta.get("reference_year"), meta.get("source", "external"))
