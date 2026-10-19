"""
app/coverage/methods.py - method coverage.

Share of the method-variation queries (sibling methods / datasets /
modalities) that returned a relevant item.  None when no sibling exists.
"""

from __future__ import annotations


def compute_method_coverage(items: list[dict], method_queries: list[str]) -> float | None:
    if not method_queries:
        return None
    hit = {f["query"] for it in items if it["evidence_type"] != "Insufficient" for f in it.get("found_by", [])
           if f["search_type"] == "method_variation"}
    return round(sum(1 for q in method_queries if q in hit) / len(method_queries), 4)
