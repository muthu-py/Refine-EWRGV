"""
app/coverage/terminology.py - terminology coverage.

Share of the terminology (synonym / alias) queries that returned a relevant
item.  None when the gap's concepts have no aliases to try.
"""

from __future__ import annotations


def compute_terminology_coverage(items: list[dict], terminology_queries: list[str]) -> float | None:
    if not terminology_queries:
        return None
    hit = {f["query"] for it in items if it["evidence_type"] != "Insufficient" for f in it.get("found_by", [])
           if f["search_type"] == "terminology"}
    return round(sum(1 for q in terminology_queries if q in hit) / len(terminology_queries), 4)
