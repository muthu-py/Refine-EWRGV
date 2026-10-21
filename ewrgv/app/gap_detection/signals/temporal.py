"""
app/gap_detection/signals/temporal.py
---------------------------------------
D3 - temporal staleness tau (the ResearchRAG beta-term, made concrete).

Internal staleness, from the run corpus (on-topic papers):

    tau_int(x) = 0.6 * min(1, (Y_ref - Y_last(x)) / 5) + 0.4 * sigmoid(-20 * slope(x))

    Y_ref     latest publication year in the corpus
    Y_last    latest year a paper mentions x
    slope     least-squares slope of x's yearly share of papers (last 5 years)

External staleness, from OpenAlex counts per publication year of
'"x" AND "<anchor>"' divided by those of '"<anchor>"' (the field, e.g.
"retinal"), over the last 6 complete years:

    rel_slope = slope(share) / mean(share)        (fractional change per year)
    tau_ext   = 1 / (1 + exp(8 * rel_slope))      (falling share -> towards 1)
                = 1.0 if x has no external paper in the last 2 complete years

Combined per concept: weighted mean with w_ext = 1 and
w_int = 0.5 * min(1, corpus_span_years / 4) (a 5-year corpus says little about
trends).  A candidate's tau is the mean over its core concepts, raised to
>= 0.8 by the stale-future-work rule (newest statement >= 3 years older than
the corpus and no later paper addressing it).

D3 also emits candidates of type Temporal for specific concepts with
tau >= 0.65 and >= 3 papers.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Optional

from app.gap_detection.signals.explicit import RawCandidate
from app.knowledge_graph.graph import KnowledgeGraph
from app.validation.sources import phrase_query

DETECTOR = "temporal_staleness"
TREND_TYPES = {"METHOD", "DATASET", "MODALITY", "TASK", "DISEASE", "CONCEPT", "POPULATION"}
STALE_THRESHOLD = 0.65
MAX_EXTERNAL_CONCEPTS = 25


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    den = sum((x - mx) ** 2 for x in xs)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den if den else 0.0


def corpus_years(kg: KnowledgeGraph) -> tuple[Optional[int], Optional[int]]:
    years = [kg.papers[p]["year"] for p in kg.on_topic_papers if kg.papers[p].get("year")]
    return (min(years), max(years)) if years else (None, None)


def internal_staleness(kg: KnowledgeGraph, eid: str) -> Optional[dict]:
    y_min, y_ref = corpus_years(kg)
    if y_ref is None:
        return None
    papers_by_year: dict[int, int] = {}
    for p in kg.on_topic_papers:
        y = kg.papers[p].get("year")
        if y:
            papers_by_year[y] = papers_by_year.get(y, 0) + 1
    years = [kg.papers[p].get("year") for p in kg.entities[eid]["papers"] if kg.papers.get(p, {}).get("year")]
    if not years:
        return None
    y_last = max(years)
    window = [y for y in range(y_ref - 4, y_ref + 1) if papers_by_year.get(y)]
    shares = [sum(1 for v in years if v == y) / papers_by_year[y] for y in window]
    slope = _slope([float(y) for y in window], shares)
    tau = 0.6 * min(1.0, (y_ref - y_last) / 5) + 0.4 * _sigmoid(-20 * slope)
    return {"tau_internal": round(tau, 4), "last_year": y_last, "share_slope": round(slope, 4),
            "corpus_span": (y_ref - y_min) if y_min else 0}


def external_staleness(series: Optional[dict[int, int]], field_series: Optional[dict[int, int]],
                       this_year: Optional[int] = None) -> Optional[dict]:
    if not series or not field_series:
        return None
    this_year = this_year or date.today().year
    years = [y for y in range(this_year - 6, this_year) if field_series.get(y)]
    if len(years) < 3:
        return None
    shares = [series.get(y, 0) / field_series[y] for y in years]
    mean = sum(shares) / len(shares)
    if mean <= 0:
        return {"tau_external": 1.0, "external_share": 0.0, "external_rel_slope": None, "external_years": years}
    rel_slope = _slope([float(y) for y in years], shares) / mean
    recent_zero = all(series.get(y, 0) == 0 for y in years[-2:])
    tau = 1.0 if recent_zero else 1.0 / (1.0 + math.exp(8 * rel_slope))
    return {"tau_external": round(tau, 4), "external_share": round(mean, 5),
            "external_rel_slope": round(rel_slope, 4), "external_years": [years[0], years[-1]],
            "external_counts": {str(y): series.get(y, 0) for y in years}}


# Method-family terms are not a field: "machine learning" would measure trends in all of ML.
_NOT_A_FIELD = {"machine learning", "deep learning", "computer vision", "artificial intelligence", "image analysis",
                "image processing", "neural network", "disease detection", "detection", "classification"}


def field_anchor(kg: KnowledgeGraph) -> Optional[str]:
    """The field term trends are measured within (the most frequent domain term of the question)."""
    cands = [e for e in kg.entities.values()
             if e.get("query_term") and e["type"] in ("DISEASE", "MODALITY", "CONCEPT") and e["name"].lower() not in _NOT_A_FIELD]
    if not cands:
        return None
    # disease / modality terms name the field more precisely than generic concepts
    name = max(cands, key=lambda e: (e["type"] in ("DISEASE", "MODALITY"), len(e["papers"])))["name"]
    # the shortest content word of the question term keeps the field broad ("retinal disease" -> "retinal")
    words = [w for w in name.split() if len(w) > 3]
    return words[0] if words else name


async def staleness_context(kg: KnowledgeGraph, concepts: list[str], external: Any = None) -> dict:
    """tau per concept id (internal + external trends) and the anchor used."""
    y_min, y_ref = corpus_years(kg)
    span = (y_ref - y_min) if y_ref and y_min else 0
    w_int = 0.5 * min(1.0, span / 4)
    anchor = field_anchor(kg)
    field_series = None
    if external is not None and anchor:
        field_series = await external.openalex_counts_by_year(phrase_query([anchor]))
    out: dict[str, dict] = {}
    for i, eid in enumerate(dict.fromkeys(concepts)):
        if eid not in kg.entities:
            continue
        info = internal_staleness(kg, eid) or {}
        ext = None
        if external is not None and anchor and field_series and i < MAX_EXTERNAL_CONCEPTS:
            name = kg.entities[eid]["name"]
            terms = [name] if anchor.lower() in name.lower() else [name, anchor]
            ext = external_staleness(await external.openalex_counts_by_year(phrase_query(terms)), field_series)
        if ext:
            info.update(ext)
        parts = []
        if "tau_internal" in info:
            parts.append((w_int, info["tau_internal"]))
        if ext:
            parts.append((1.0, ext["tau_external"]))
        total = sum(w for w, _ in parts)
        info["tau"] = round(sum(w * v for w, v in parts) / total, 4) if total else None
        info["weights"] = {"internal": round(w_int, 3), "external": 1.0 if ext else 0.0}
        out[eid] = info
    return {"by_concept": out, "anchor": anchor, "corpus_span": span, "reference_year": y_ref}


def candidate_tau(context: dict, core: list[str], signals: dict) -> Optional[float]:
    values = [context["by_concept"][e]["tau"] for e in core
              if e in context["by_concept"] and context["by_concept"][e].get("tau") is not None]
    tau = sum(values) / len(values) if values else None
    d1 = signals.get("explicit_statements", {})
    latest, ref = d1.get("latest_statement_year"), context.get("reference_year")
    if latest and ref and ref - latest >= 3 and not d1.get("addressed_later"):
        tau = max(tau or 0.0, 0.8)  # stale future work: stated long ago, nobody followed up
    return None if tau is None else round(tau, 4)


def detect_stale(kg: KnowledgeGraph, context: dict, min_papers: int = 3) -> list[RawCandidate]:
    out = []
    for eid, info in context["by_concept"].items():
        e = kg.entities.get(eid)
        if not e or e.get("generic") or e.get("query_term") or e["type"] not in TREND_TYPES:
            continue
        if len(e["papers"]) < min_papers or (info.get("tau") or 0) < STALE_THRESHOLD:
            continue
        trend = info.get("external_rel_slope")
        trend_txt = f"its share of {context['anchor']} papers changes {trend * 100:+.0f}% per year (OpenAlex)" if trend is not None else \
            f"it was last seen in {info.get('last_year')} in this corpus"
        statement = f"Research on {e['name']} appears to be declining: {trend_txt}."
        chunks = kg.chunks_mentioning([eid])[:3]
        out.append(RawCandidate(
            detector=DETECTOR, statement=statement, core_entities=[eid],
            evidence_chunks=[(c, f"mentions {e['name']}") for c in chunks],
            signals={"tau": info["tau"], **{k: v for k, v in info.items() if k != "tau"}},
        ))
    return out
