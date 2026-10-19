"""
app/gap_detection/signals/explicit.py
---------------------------------------
D1 - explicit-statement consensus.

Gap statements extracted into the knowledge graph (limitations, future work,
open problems) are clustered by meaning across papers; each cluster becomes
one candidate gap.  The more independent papers state it, and the stronger
their cues, the stronger the explicit signal:

    support = sum over papers of the strongest statement strength in the cluster
    E       = 1 - exp(-support / 2)

Early counter-evidence ("addressed_later"): a paper published after the
newest statement whose Methods/Results/Abstract chunks mention all of the
gap's core concepts.  It is reported, not scored (validation decides).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from app.knowledge_graph.graph import KnowledgeGraph

DETECTOR = "explicit_statements"
CLUSTER_SIMILARITY = 0.72        # average-linkage cosine; statements this similar state the same gap
_ADDRESSING_SECTIONS = {"METHODS", "METHODOLOGY", "MODEL", "RESULTS", "EXPERIMENTS", "EVALUATION", "ABSTRACT"}


@dataclass
class RawCandidate:
    """Detector output before fusion and scoring."""
    detector: str
    statement: str
    core_entities: list[str]
    statements: list[dict] = field(default_factory=list)      # supporting gap statements
    evidence_chunks: list[tuple[str, str]] = field(default_factory=list)  # (chunk_id, role)
    counter_chunks: list[tuple[str, str]] = field(default_factory=list)   # (chunk_id, reason)
    signals: dict[str, Any] = field(default_factory=dict)


def _clean(sentence: str) -> str:
    s = sentence.strip()
    for lead in ("However, ", "Furthermore, ", "Moreover, ", "In addition, ", "Additionally, ", "Also, ",
                 "Nevertheless, ", "Still, ", "Unfortunately, ", "Notably, ", "Overall, "):
        if s.startswith(lead):
            s = s[len(lead):]
            s = s[:1].upper() + s[1:]
    return s


def core_entities(kg: KnowledgeGraph, statements: list[dict], limit: int = 4) -> list[str]:
    """Specific entities shared by the statements (generic and question terms excluded)."""
    counts: dict[str, int] = {}
    for s in statements:
        for eid in set(s["entities"]):
            e = kg.entities.get(eid)
            if e is None or e.get("generic") or e.get("query_term"):
                continue
            counts[eid] = counts.get(eid, 0) + 1
    if not counts:
        return []
    need = max(1, math.ceil(len(statements) * 0.5)) if len(statements) > 1 else 1
    shared = [e for e, n in counts.items() if n >= need]
    ranked = sorted(shared or counts, key=lambda e: (-counts[e], -len(kg.entities[e]["papers"]), e))
    return ranked[:limit]


def _specific(kg: KnowledgeGraph, statement: dict) -> bool:
    """True when the statement names at least one specific concept (not generic, not the question's topic)."""
    for eid in statement["entities"]:
        e = kg.entities.get(eid)
        if e and not e.get("generic") and not e.get("query_term"):
            return True
    return False


def detect_explicit(kg: KnowledgeGraph, embedder: Any, similarity: float = CLUSTER_SIMILARITY) -> list[RawCandidate]:
    # A statement that names no specific concept ("several challenges remain") cannot define a gap.
    statements = [
        s for s in kg.statements
        if kg.papers.get(s["paper_id"], {}).get("on_topic", True) and _specific(kg, s)
    ]
    if not statements:
        return []
    vectors = np.asarray(embedder.embed_batch([s["sentence"] for s in statements]), dtype=float)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    if len(statements) == 1:
        labels = np.zeros(1, dtype=int)
    else:
        labels = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average", distance_threshold=1.0 - similarity,
        ).fit_predict(vectors)

    groups = [[i for i, l in enumerate(labels) if l == label] for label in sorted(set(labels.tolist()))]
    groups = _merge_by_concept(kg, statements, vectors, groups)

    candidates: list[RawCandidate] = []
    for idx in groups:
        members = [statements[i] for i in idx]
        sub = vectors[idx]
        centroid = sub.mean(axis=0)
        sims = sub @ centroid
        # representative: central, strongly cued, and a well-formed sentence
        medoid = max(range(len(idx)), key=lambda k: float(sims[k]) + 0.1 * members[k]["strength"]
                     + _form_bonus(members[k]["sentence"]))
        per_paper: dict[str, float] = {}
        for s in members:
            per_paper[s["paper_id"]] = max(per_paper.get(s["paper_id"], 0.0), s["strength"])
        support = sum(per_paper.values())
        years = [s["year"] for s in members if s.get("year")]
        core = core_entities(kg, members)
        cue_mix: dict[str, int] = {}
        for s in members:
            cue_mix[s["cue_class"]] = cue_mix.get(s["cue_class"], 0) + 1
        cand = RawCandidate(
            detector=DETECTOR,
            statement=_clean(members[medoid]["sentence"]),
            core_entities=core,
            statements=sorted(members, key=lambda s: -s["strength"]),
            evidence_chunks=[(s["chunk_id"], "gap_statement") for s in members],
            signals={
                "statements": len(members),
                "stating_papers": len(per_paper),
                "support": round(support, 4),
                "explicit": round(1 - math.exp(-support / 2), 4),
                "cue_mix": cue_mix,
                "latest_statement_year": max(years) if years else None,
                "cluster_cohesion": round(float(sims.mean()), 4),
            },
        )
        cand.counter_chunks = _addressed_later(kg, core, set(per_paper), max(years) if years else None)
        cand.signals["addressed_later"] = len({kg.chunk_meta[c]["paper_id"] for c, _ in cand.counter_chunks})
        candidates.append(cand)
    return candidates


CONCEPT_MERGE_JACCARD = 0.5
CONCEPT_MERGE_SIMILARITY = 0.60
NEAR_DUPLICATE_SIMILARITY = 0.82


def _form_bonus(sentence: str) -> float:
    words = len(sentence.split())
    bonus = 0.05 if sentence[:1].isupper() else -0.1
    if words > 45:
        bonus -= 0.05
    return bonus


def _merge_by_concept(kg: KnowledgeGraph, statements: list[dict], vectors, groups: list[list[int]]) -> list[list[int]]:
    """
    Second pass: clusters that are about the same specific concepts and still
    semantically close (centroid cosine >= 0.60) state the same gap in
    different words ("class imbalance" phrased five ways); merge them so the
    number of independent stating papers is not split.
    """
    if len(groups) < 2:
        return groups
    cores = [set(core_entities(kg, [statements[i] for i in g])) for g in groups]
    centroids = []
    for g in groups:
        c = vectors[g].mean(axis=0)
        centroids.append(c / (np.linalg.norm(c) + 1e-12))
    n = len(groups)
    sims = [[float(centroids[i] @ centroids[j]) for j in range(n)] for i in range(n)]

    def qualifies(i: int, j: int) -> bool:
        a, b = cores[i], cores[j]
        same_concepts = bool(a and b) and len(a & b) / len(a | b) >= CONCEPT_MERGE_JACCARD
        # Near-duplicates: same concepts and close, or very close in meaning even
        # when tagged with different concepts ("validate in diverse clinical
        # settings" vs "evaluate in real-world clinical conditions").
        return (same_concepts and sims[i][j] >= CONCEPT_MERGE_SIMILARITY) or sims[i][j] >= NEAR_DUPLICATE_SIMILARITY

    # Complete linkage over groups: two merged sets join only if EVERY cross pair
    # qualifies, so A~B and B~C never chains A with an unrelated C.
    cluster_of = list(range(n))
    members: dict[int, set[int]] = {i: {i} for i in range(n)}
    pairs = sorted(((sims[i][j], i, j) for i in range(n) for j in range(i + 1, n) if qualifies(i, j)), reverse=True)
    for _, i, j in pairs:
        ci, cj = cluster_of[i], cluster_of[j]
        if ci == cj:
            continue
        if all(qualifies(a, b) for a in members[ci] for b in members[cj]):
            for k in members[cj]:
                cluster_of[k] = ci
            members[ci] |= members.pop(cj)
    return [[idx for g in sorted(m) for idx in groups[g]] for m in members.values()]


def _addressed_later(kg: KnowledgeGraph, core: list[str], stating: set[str], latest: int | None) -> list[tuple[str, str]]:
    if len(core) < 2 or latest is None:
        return []
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for cid in kg.chunks_mentioning(core):
        meta = kg.chunk_meta.get(cid, {})
        pid = meta.get("paper_id")
        year = kg.papers.get(pid, {}).get("year")
        if pid in stating or pid in seen or not year or year <= latest:
            continue
        if meta.get("section_type") not in _ADDRESSING_SECTIONS:
            continue
        seen.add(pid)
        out.append((cid, f"published {year}, after the newest statement ({latest}); mentions all core concepts in {meta.get('section_type')}"))
    return out[:5]
