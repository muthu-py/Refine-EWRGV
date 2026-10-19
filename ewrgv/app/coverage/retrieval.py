"""
app/coverage/retrieval.py - retrieval coverage.

Share of the verification queries that returned at least one relevant item
(any evidence type other than INSUFFICIENT).  Low retrieval coverage means
the searches did not reach the literature the gap is about, so the absence
of counter evidence says little.
"""

from __future__ import annotations


def compute_retrieval_coverage(items: list[dict], queries: list[str]) -> float | None:
    if not queries:
        return None
    hit_queries = {f["query"] for it in items if it["evidence_type"] != "Insufficient" for f in it.get("found_by", [])}
    return round(sum(1 for q in queries if q in hit_queries) / len(queries), 4)
