"""
app/evidence/aggregator.py
---------------------------
Evidence aggregator: merges hits collected by the direct, terminology,
method-variation and counter searches (and later iterations) into one
de-duplicated, weighted evidence list.

* The same work found several times (same DOI, same normalised title, or the
  same corpus chunk) is one item; it keeps its strongest classification
  (CONTRADICTORY > COUNTER > PARTIAL > SUPPORTING > INSUFFICIENT on ties of
  strength) and records every search type / query that found it.
* A corpus chunk and an external record of the same paper (same DOI) are
  also merged, so a paper never counts twice.
"""

from __future__ import annotations

import re
from typing import Optional

from app.domain.enums import EvidenceType
from app.domain.models.evidence import Evidence

_PRIORITY = {
    EvidenceType.CONTRADICTORY.value: 4, EvidenceType.COUNTER.value: 3, EvidenceType.PARTIAL.value: 2,
    EvidenceType.SUPPORTING.value: 1, EvidenceType.INSUFFICIENT.value: 0,
}


def _key(item: dict) -> str:
    if item.get("doi"):
        return "doi:" + item["doi"].lower().strip()
    if item.get("source") == "corpus" and item.get("paper_id"):
        return "paper:" + item["paper_id"]
    title = re.sub(r"[^a-z0-9]", "", (item.get("title") or "").lower())
    return "title:" + title if title else "id:" + str(item.get("id"))


class EvidenceAggregator:
    def aggregate_items(self, items: list[dict]) -> list[dict]:
        """De-duplicate classified hits (dicts with evidence_type and strength)."""
        best: dict[str, dict] = {}
        for it in items:
            k = _key(it)
            found = {"search_type": it["search_type"], "query": it["query"], "source": it["source"], "rank": it.get("rank")}
            if k not in best:
                best[k] = {**it, "found_by": [found]}
                continue
            cur = best[k]
            cur["found_by"].append(found)
            if (_PRIORITY[it["evidence_type"]], it["strength"]) > (_PRIORITY[cur["evidence_type"]], cur["strength"]):
                keep = cur["found_by"]
                best[k] = {**it, "found_by": keep}
        return sorted(best.values(), key=lambda x: (-_PRIORITY[x["evidence_type"]], -x["strength"]))

    def aggregate(self, evidence_items: list[Evidence]) -> list[Evidence]:
        """Domain-model variant: keep the strongest Evidence per passage."""
        best: dict[str, Evidence] = {}
        for ev in evidence_items:
            k = ev.passage_id or ev.paper_id
            if k not in best or ev.strength > best[k].strength:
                best[k] = ev
        return list(best.values())


def to_evidence(gap_id: str, item: dict, reference_year: Optional[int]) -> Evidence:
    """Convert an aggregated item into the domain Evidence model."""
    from app.evidence.provenance import attach_provenance

    ev = Evidence(
        gap_id=gap_id, paper_id=item.get("paper_id") or item.get("doi") or item.get("id") or "",
        passage_id=str(item.get("id") or ""), evidence_type=EvidenceType(item["evidence_type"]),
        passage_text=item["text"][:1500], relevance=item["relevance"], strength=item["strength"],
        metadata={k: item.get(k) for k in ("source", "title", "year", "doi", "url", "section", "section_type",
                                           "pages", "concept_coverage", "concepts_named", "rationale")} | {"reference_year": reference_year},
    )
    first = item["found_by"][0]
    return attach_provenance(ev, first["search_type"], first["query"], first.get("rank"))
