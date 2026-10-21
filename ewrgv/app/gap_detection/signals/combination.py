"""
app/gap_detection/signals/combination.py
------------------------------------------
D2 - combination sparsity with link prediction on the knowledge graph.

For entity pairs of complementary types (method x modality, method x
disease, ...), each well represented in the run:

    observed  = papers mentioning both            n_AB
    expected  = n_A * n_B / N                     (independence)
    lift      = observed / expected
    sparsity  = 1 - min(lift, 1)

A sparse cell is only interesting when the two concepts *should* meet: they
share many co-occurrence neighbours in the graph (link prediction, Jaccard
of neighbourhoods).  "Common A, common B, similar contexts, almost never
together" is the ResearchRAG density idea without the GNN.

The run corpus is small, so D2 candidates are hypotheses: the papers that
do combine A and B are reported as counter hints, and EWRGV validation must
check external coverage before calling them gaps.
"""

from __future__ import annotations

from typing import Any

from app.gap_detection.signals.explicit import RawCandidate
from app.knowledge_graph.graph import KnowledgeGraph

DETECTOR = "combination_sparsity"

TYPE_PAIRS: list[tuple[str, str]] = [
    ("METHOD", "MODALITY"),
    ("METHOD", "DISEASE"),
    ("METHOD", "DATASET"),
    ("METHOD", "TASK"),
    ("METHOD", "POPULATION"),
    ("MODALITY", "DISEASE"),
    ("DISEASE", "POPULATION"),
    ("METHOD", "CONCEPT"),
]
_TEMPLATES = {
    ("METHOD", "DATASET"): "{a} is rarely evaluated on the {b} dataset",
    ("METHOD", "POPULATION"): "{a} is rarely studied in {b}",
    ("DISEASE", "POPULATION"): "{a} is rarely studied in {b}",
}
_DEFAULT_TEMPLATE = "{a} is rarely combined with {b}"


def _neighbours(kg: KnowledgeGraph, min_papers: int) -> dict[str, set[str]]:
    nbrs: dict[str, set[str]] = {e: set() for e in kg.entities}
    for edge in kg.cooccurrence:
        if edge["papers"] >= min_papers and edge["pmi"] > 0:
            nbrs[edge["a"]].add(edge["b"])
            nbrs[edge["b"]].add(edge["a"])
    return nbrs


def detect_combinations(
    kg: KnowledgeGraph,
    min_papers: int = 3,
    max_lift: float = 0.35,
    min_expected: float = 1.5,
    min_link: float = 0.25,
    limit: int = 15,
) -> list[RawCandidate]:
    n = max(1, len(kg.on_topic_papers))
    usable = {
        eid: e for eid, e in kg.entities.items()
        if len(e["papers"]) >= min_papers and not e.get("generic") and not e.get("query_term")
    }
    nbrs = _neighbours(kg, min_papers=2)
    pairs = {tuple(sorted((ta, tb))): (ta, tb) for ta, tb in TYPE_PAIRS}
    by_type: dict[str, list[str]] = {}
    for eid, e in usable.items():
        by_type.setdefault(e["type"], []).append(eid)

    found: list[tuple[float, RawCandidate]] = []
    for ta, tb in pairs.values():
        for a in by_type.get(ta, []):
            for b in by_type.get(tb, []):
                if a == b:
                    continue
                pa, pb = set(usable[a]["papers"]), set(usable[b]["papers"])
                observed = len(pa & pb)
                expected = len(pa) * len(pb) / n
                if expected < min_expected:
                    continue
                lift = observed / expected
                if lift > max_lift:
                    continue
                na, nb = nbrs.get(a, set()) - {b}, nbrs.get(b, set()) - {a}
                union = na | nb
                link = len(na & nb) / len(union) if union else 0.0
                if link < min_link:
                    continue
                sparsity = 1 - min(lift, 1.0)
                statement = _TEMPLATES.get((ta, tb), _DEFAULT_TEMPLATE).format(a=usable[a]["name"], b=usable[b]["name"])
                statement = (f"{statement[0].upper()}{statement[1:]}: {len(pa)} papers study {usable[a]['name']} "
                             f"and {len(pb)} study {usable[b]['name']}, but only {observed} combine them "
                             f"(expected {expected:.1f}).")
                cand = RawCandidate(
                    detector=DETECTOR,
                    statement=statement,
                    core_entities=[a, b],
                    evidence_chunks=[(c, f"mentions {usable[a]['name']}") for c in _top_chunks(kg, a, pa - pb)]
                    + [(c, f"mentions {usable[b]['name']}") for c in _top_chunks(kg, b, pb - pa)],
                    counter_chunks=[(c, f"combines {usable[a]['name']} and {usable[b]['name']}")
                                    for c in kg.chunks_mentioning([a, b])[:3]],
                    signals={
                        "papers_a": len(pa), "papers_b": len(pb), "papers_both": observed,
                        "expected": round(expected, 3), "lift": round(lift, 4),
                        "sparsity": round(sparsity, 4), "link_prediction": round(link, 4),
                        "shared_neighbours": len(na & nb), "pair_types": [ta, tb],
                    },
                )
                found.append((sparsity * (0.5 + 0.5 * link), cand))
    found.sort(key=lambda x: -x[0])
    return [c for _, c in found[:limit]]


def _top_chunks(kg: KnowledgeGraph, eid: str, papers: set[str], k: int = 2) -> list[str]:
    """Up to k chunks mentioning eid, from different papers."""
    out, seen = [], set()
    for cid in kg.chunks_mentioning([eid]):
        pid = kg.chunk_meta.get(cid, {}).get("paper_id")
        if pid in papers and pid not in seen:
            seen.add(pid)
            out.append(cid)
        if len(out) >= k:
            break
    return out


def pair_sparsity(kg: KnowledgeGraph, core: list[str]) -> tuple[float, dict] | None:
    """Sparsity of the sparsest pairable entity pair among ``core`` (for fusing D1 candidates)."""
    n = max(1, len(kg.on_topic_papers))
    allowed = {tuple(sorted(p)) for p in TYPE_PAIRS}
    best = None
    for i, a in enumerate(core):
        for b in core[i + 1:]:
            ea, eb = kg.entities.get(a), kg.entities.get(b)
            if not ea or not eb or tuple(sorted((ea["type"], eb["type"]))) not in allowed:
                continue
            pa, pb = set(ea["papers"]), set(eb["papers"])
            expected = len(pa) * len(pb) / n
            if expected <= 0:
                continue
            lift = len(pa & pb) / expected
            s = 1 - min(lift, 1.0)
            if best is None or s > best[0]:
                best = (s, {"pair": [a, b], "lift": round(lift, 4), "papers_both": len(pa & pb),
                            "expected": round(expected, 3)})
    return best
