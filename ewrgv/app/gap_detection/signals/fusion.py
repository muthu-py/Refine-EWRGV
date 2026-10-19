"""
app/gap_detection/signals/fusion.py
-------------------------------------
Fusion of detector candidates, the full GapScore, the ResearchRAG baseline
score, gap typing and ranking.

GapScore (no LLM confidence anywhere):

    E    explicit      1 - exp(-support/2) from gap statements about the candidate
    S    sparsity      max of (a) its sparsest concept pair, (1 - min(lift, 1)), for D2
                       pairs x (0.5 + 0.5 * link_prediction); (b) 1 - rho of its topic
                       (D4); discounted by corpus size: S_eff = S * min(1, N_on_topic / 50)
                       (sparsity in a small corpus is weak evidence until EWRGV checks
                       external coverage; ResearchRAG's 24% spurious gaps)
    tau  staleness     D3 (temporal.py); weight halved when only the internal trend
                       of a corpus spanning < 4 years is available
    eta  novelty       D4 novelty of the topic its evidence falls in (topics.py)
    C    contradiction D5 (phase P5, not yet available: excluded from the weights)
    I    independence  min(distinct stating papers / 3, 1)
    A    agreement     (number of detectors - 1) / 3            (bonus)

    GapScore = sum(w_k * v_k over available k) / sum(w_k over available k) + 0.10 A,
               capped at 1, with w = E .25, S .25, tau .15, eta .10, C .10, I .15

ResearchRAG baseline (Ghosh 2026, eq. 5), heuristic density, computed alongside:

    GapScore_RR = 0.45 (1 - rho) + 0.35 tau + 0.20 eta
                  (1 - rho = undiscounted sparsity; missing terms count as 0)
"""

from __future__ import annotations

import math
import re
from typing import Optional

from app.domain.enums import GapType
from app.gap_detection.signals.combination import DETECTOR as D2, pair_sparsity
from app.gap_detection.signals.explicit import DETECTOR as D1, RawCandidate
from app.gap_detection.signals.temporal import DETECTOR as D3, candidate_tau
from app.gap_detection.signals.topics import DETECTOR as D4, topic_of
from app.knowledge_graph.graph import KnowledgeGraph

WEIGHTS = {"explicit": 0.25, "sparsity": 0.25, "staleness": 0.15, "novelty": 0.10,
           "contradiction": 0.10, "independence": 0.15}
RESEARCHRAG_WEIGHTS = {"sparsity": 0.45, "staleness": 0.35, "novelty": 0.20}
AGREEMENT_WEIGHT = 0.10
SMALL_CORPUS_PAPERS = 50
INDEPENDENCE_SATURATION = 3
DIVERSITY_PENALTY = 0.05
DIVERSITY_CAP = 2          # at most 2 x penalty, so strong gaps of a common type are not buried

_POPULATION_RE = re.compile(r"\b(population|ethnic|demographic|children|paediatric|pediatric|elderly|rural|low-resource|underserved|countr|region|geograph)", re.I)
_DATASET_RE = re.compile(r"\b(dataset|data set|labell?ed data|annotat|benchmark data|data scarcity|small data|data imbalance|class imbalance)", re.I)
_EVAL_RE = re.compile(r"\b(evaluat|validat|benchmark|metric|external test|comparison|compared)", re.I)
_CONTRA_RE = re.compile(r"\b(contradict|conflict|inconsistent|disagree)", re.I)


def gap_type_for(kg: KnowledgeGraph, core: list[str], statement: str, detectors: Optional[list[str]] = None) -> GapType:
    types = {kg.entities[e]["type"] for e in core if e in kg.entities}
    if detectors and detectors[0] == D3:
        return GapType.TEMPORAL
    if _CONTRA_RE.search(statement):
        return GapType.CONTRADICTORY_EVIDENCE
    if "POPULATION" in types or _POPULATION_RE.search(statement):
        return GapType.POPULATION
    if "DATASET" in types or _DATASET_RE.search(statement):
        return GapType.DATASET
    if "METRIC" in types or _EVAL_RE.search(statement):
        return GapType.EVALUATION
    return GapType.METHODOLOGICAL


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def fuse(kg: KnowledgeGraph, candidates: list[RawCandidate]) -> list[dict]:
    """
    Merge candidates that describe the same gap.

    D2 pairs / D3 concepts / D4 topics join an explicit (D1) candidate when
    they are about its concepts (or, for D4, its evidence lies in the topic);
    otherwise they stand alone and pick up any gap statements about their
    concepts.  Agreement between detectors is recorded for the A bonus.
    """
    merged: list[dict] = [{"primary": c, "detectors": [D1], "extra": []} for c in candidates if c.detector == D1]
    for sc in [c for c in candidates if c.detector != D1]:
        concepts = set(sc.core_entities)
        host = None
        for m in merged:
            if m["primary"].detector != D1:
                continue
            core = set(m["primary"].core_entities)
            if sc.detector == D2 and (concepts <= core or (len(core) >= 2 and _jaccard(concepts, core) >= 0.6)):
                host = m
            elif sc.detector == D3 and core and m["primary"].core_entities[0] in concepts:
                host = m
            elif sc.detector == D4:
                chunks = {c for c, _ in sc.evidence_chunks}
                host_chunks = {c for c, _ in m["primary"].evidence_chunks}
                if len(chunks & host_chunks) >= 2 or (core and concepts and _jaccard(concepts, core) >= 0.5):
                    host = m
            if host is not None:
                break
        if host is not None:
            if sc.detector not in host["detectors"]:
                host["detectors"].append(sc.detector)
            host["extra"].append(sc)
            continue
        if concepts:
            # A structural gap that authors also state explicitly gains those statements.
            need = concepts if sc.detector == D2 else set(list(concepts)[:1])
            stated = [s for s in kg.statements if need <= set(s["entities"])]
            sc.statements = sorted(stated, key=lambda s: -s["strength"])[:10]
            sc.evidence_chunks += [(s["chunk_id"], "gap_statement") for s in sc.statements]
        merged.append({"primary": sc, "detectors": [sc.detector], "extra": []})
    return merged


def score(kg: KnowledgeGraph, group: dict, context: Optional[dict] = None) -> dict:
    """Full GapScore + ResearchRAG baseline, with a transparent breakdown."""
    context = context or {}
    primary: RawCandidate = group["primary"]
    sources = [primary, *group["extra"]]
    statements = primary.statements
    per_paper: dict[str, float] = {}
    for s in statements:
        per_paper[s["paper_id"]] = max(per_paper.get(s["paper_id"], 0.0), s["strength"])
    support = sum(per_paper.values())
    explicit = 1 - math.exp(-support / 2) if support else 0.0

    # --- sparsity: concept pairs (D2 / pair lift) and topic density (D4)
    sparsity_raw, sparsity_detail = 0.0, None
    for c in sources:
        if c.detector == D2:
            plausible = c.signals["sparsity"] * (0.5 + 0.5 * c.signals["link_prediction"])
            if plausible > sparsity_raw:
                sparsity_raw, sparsity_detail = plausible, {"source": "concept_pair", **{k: c.signals[k] for k in ("lift", "papers_both", "expected", "link_prediction")}}
    if sparsity_detail is None and len(primary.core_entities) >= 2:
        best = pair_sparsity(kg, primary.core_entities)
        if best is not None:
            sparsity_raw, sparsity_detail = best[0], {"source": "concept_pair", **best[1]}
    topic = None
    for c in sources:
        if c.detector == D4:
            topic = {"id": c.signals["topic_id"], "label": c.signals["topic_label"], "density": c.signals["density"],
                     "novelty": c.signals["novelty"]}
    if topic is None and context.get("topics"):
        t = topic_of(context["topics"], [cid for cid, _ in primary.evidence_chunks])
        if t is not None:
            topic = {"id": t["id"], "label": t["label"], "density": t["density"], "novelty": t["novelty"]}
    if topic is not None and 1 - topic["density"] > sparsity_raw:
        sparsity_raw, sparsity_detail = 1 - topic["density"], {"source": "topic_density", "topic": topic["label"],
                                                               "density": topic["density"]}
    n_on = len(kg.on_topic_papers)
    reliability = min(1.0, n_on / SMALL_CORPUS_PAPERS)
    sparsity = sparsity_raw * reliability

    # --- staleness (D3) and novelty (D4)
    signals_by_detector = {c.detector: c.signals for c in sources}
    staleness = None
    staleness_reliability = 1.0
    if context.get("staleness"):
        staleness = candidate_tau(context["staleness"], primary.core_entities, signals_by_detector)
        has_external = any(context["staleness"]["by_concept"].get(e, {}).get("tau_external") is not None
                           for e in primary.core_entities)
        if staleness is not None and not has_external and context["staleness"].get("corpus_span", 0) < 4:
            staleness_reliability = 0.5
    for c in sources:
        if c.detector == D3:
            staleness = max(staleness or 0.0, c.signals["tau"])
    novelty = topic["novelty"] if topic is not None else None
    independence = min(len(per_paper) / INDEPENDENCE_SATURATION, 1.0)
    agreement = min((len(set(group["detectors"])) - 1) / 3, 1.0)

    values = {"explicit": explicit, "sparsity": sparsity, "staleness": staleness, "novelty": novelty,
              "contradiction": None, "independence": independence}
    weights = {k: WEIGHTS[k] * (staleness_reliability if k == "staleness" else 1.0)
               for k, v in values.items() if v is not None}
    total_w = sum(weights.values())
    breakdown = {}
    for name, v in values.items():
        if v is None:
            breakdown[name] = {"value": None, "weight": 0.0, "contribution": 0.0,
                               "note": "not available (phase P5)" if name == "contradiction" else "not available"}
        else:
            w = weights[name] / total_w
            breakdown[name] = {"value": round(v, 4), "weight": round(w, 4), "contribution": round(w * v, 4)}
    breakdown["agreement_bonus"] = {"value": round(agreement, 4), "weight": AGREEMENT_WEIGHT,
                                    "contribution": round(AGREEMENT_WEIGHT * agreement, 4)}
    gap_score = min(1.0, sum(b["contribution"] for b in breakdown.values()))
    rr = (RESEARCHRAG_WEIGHTS["sparsity"] * sparsity_raw + RESEARCHRAG_WEIGHTS["staleness"] * (staleness or 0.0)
          + RESEARCHRAG_WEIGHTS["novelty"] * (novelty or 0.0))
    return {
        "gap_score": round(gap_score, 4),
        "researchrag_score": round(rr, 4),
        "score_breakdown": breakdown,
        "score_notes": {
            "support": round(support, 4),
            "stating_papers": len(per_paper),
            "sparsity_raw": round(sparsity_raw, 4),
            "sparsity_reliability": round(reliability, 4),
            "sparsity_detail": sparsity_detail,
            "staleness_reliability": staleness_reliability,
            "topic": topic,
            "on_topic_papers": n_on,
        },
    }


def rank(candidates: list[dict]) -> list[dict]:
    """Greedy ranking by gap_score with a small penalty per already-ranked candidate of the same type.

    Also sets ``researchrag_rank``: the rank by the ResearchRAG baseline score."""
    remaining = sorted(candidates, key=lambda c: (-c["gap_score"], c["gap_statement"]))
    counts: dict[str, int] = {}
    ranked: list[dict] = []
    while remaining:
        def adjusted(c: dict) -> float:
            return c["gap_score"] - DIVERSITY_PENALTY * min(counts.get(c["gap_type"], 0), DIVERSITY_CAP)

        best = max(remaining, key=lambda c: (adjusted(c), c["gap_score"], c["gap_statement"]))
        remaining.remove(best)
        penalty = best["gap_score"] - adjusted(best)
        counts[best["gap_type"]] = counts.get(best["gap_type"], 0) + 1
        best["rank"] = len(ranked) + 1
        best["rank_score"] = round(best["gap_score"] - penalty, 4)
        ranked.append(best)
    by_rr = sorted(ranked, key=lambda c: (-c.get("researchrag_score", 0.0), c["rank"]))
    for i, c in enumerate(by_rr, start=1):
        c["researchrag_rank"] = i
    return ranked


def representation(kg: KnowledgeGraph, core: list[str], statement: str, domain: Optional[str]) -> dict:
    by_type: dict[str, list[str]] = {}
    for eid in core:
        e = kg.entities.get(eid)
        if e:
            by_type.setdefault(e["type"], []).append(e["name"])
    synonyms: list[str] = []
    for eid in core:
        for alias in kg.entities.get(eid, {}).get("aliases", []):
            if alias.lower() not in {s.lower() for s in synonyms} and alias.lower() != kg.entities[eid]["name"].lower():
                synonyms.append(alias)
    return {
        "gap_statement": statement,
        "domain": domain,
        "population": ", ".join(by_type.get("POPULATION", []) + by_type.get("DISEASE", [])) or None,
        "methodology": ", ".join(by_type.get("METHOD", [])) or None,
        "outcome": ", ".join(by_type.get("TASK", []) + by_type.get("METRIC", [])) or None,
        "temporal_scope": None,
        "geographic_scope": None,
        "key_concepts": [kg.entities[e]["name"] for e in core if e in kg.entities],
        "synonyms": synonyms[:12],
    }
