"""
app/coverage/concepts.py - concept coverage.

Share of the gap's concepts named (by name or KG alias) in at least one
relevant evidence item.  A concept no relevant item mentions was not really
searched for.
"""

from __future__ import annotations


def compute_concept_coverage(concepts: list[str], items: list[dict]) -> float | None:
    if not concepts:
        return None
    named = {n for it in items if it["evidence_type"] != "Insufficient" for n in it.get("concepts_named", [])}
    return round(sum(1 for c in concepts if c in named) / len(concepts), 4)
