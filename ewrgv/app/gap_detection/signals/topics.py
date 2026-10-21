"""
app/gap_detection/signals/topics.py
-------------------------------------
D4 - topic density rho and novelty eta (the ResearchRAG alpha- and
gamma-terms, heuristic version: their edge-count baseline, no GNN).

Topics are HDBSCAN clusters of the stored chunk embeddings (on-topic,
non-noise chunks, PCA-reduced to 10 dimensions).  In a small, homogeneous
corpus chunk clusters degenerate (chunks cluster by paper: one topic holding
most papers plus single-paper topics).  When fewer than 3 topics have >= 2
papers, or one topic holds > 60% of the papers, D4 falls back to concept
clusters of the knowledge graph (Louvain over specific concepts, resolution
1.3) - ResearchRAG's unit of density.  A concept topic's papers are those
mentioning >= 2 of its concepts; its centroid is the mean of those chunks'
vectors.  ``topic_method`` in the result says which was used.  Per topic T:

    rho(T) = papers(T) / median papers per topic           (clipped to [0, 1])
    eta(T) = 0.5 * clip((1 - max cos(centroid_T, centroid_L)) / 0.3, 0, 1)
           + 0.5 * share of T's papers from the last 2 corpus years
             (L ranges over the other topics with >= median papers)

Candidates: sparse topics (rho <= 0.35) with >= 2 papers whose centroid is
relevant to the question (cosine >= median topic relevance - 0.05).  Every
other candidate gets eta (and rho) from the topic most of its evidence
chunks fall in.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Optional

import numpy as np

from app.gap_detection.signals.explicit import RawCandidate
from app.knowledge_graph.graph import KnowledgeGraph

DETECTOR = "topic_density"
MIN_CLUSTER_SIZE = 10
MIN_SAMPLES = 5
PCA_DIMENSIONS = 10
SPARSE_DENSITY = 0.35
CONCEPT_RESOLUTION = 1.3

_STOP = frozenset("""the and for with that this from are were was have has been into their which these those using used
based also than more most such other between within while where when they them both each over under about after
before during among across through results result study studies paper papers model models method methods data
images image proposed approach approaches performance accuracy table figure using shown show shows however""".split())


def _norm(m: np.ndarray) -> np.ndarray:
    return m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-12)


def build_topics(kg: KnowledgeGraph, corpus: Any, query_vector: Optional[list[float]]) -> dict:
    """Topics (HDBSCAN on chunk vectors, or KG concept clusters as fallback) + chunk -> topic map."""
    ids, vecs = [], []
    for cid in kg.chunk_meta:
        v = corpus.get_embedding(cid) if hasattr(corpus, "get_embedding") else None
        if v:
            ids.append(cid)
            vecs.append(v)
    vectors = {cid: v for cid, v in zip(ids, _norm(np.asarray(vecs, dtype=float)))} if ids else {}
    groups: dict[int, list[str]] = {}
    method = "hdbscan"
    noise = len(ids)
    if len(ids) >= MIN_CLUSTER_SIZE * 2:
        from sklearn.cluster import HDBSCAN
        from sklearn.decomposition import PCA

        x = np.asarray([vectors[c] for c in ids])
        z = PCA(n_components=min(PCA_DIMENSIONS, x.shape[1], len(ids) - 1), random_state=0).fit_transform(x)
        labels = HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, min_samples=MIN_SAMPLES, copy=True).fit_predict(z)
        for cid, label in zip(ids, labels):
            if label >= 0:
                groups.setdefault(int(label), []).append(cid)
        noise = int((labels < 0).sum())
    if _degenerate(kg, groups):
        method = "kg_concept_clusters"
        groups, noise = _concept_groups(kg), None
    if not groups:
        return {"topics": [], "chunk_topic": {}, "noise_chunks": noise, "topic_method": method}
    _, y_ref = _years(kg)

    doc_freq = Counter()
    group_terms: dict[int, Counter] = {}
    for t, chunk_ids in groups.items():
        terms = Counter()
        for cid in chunk_ids:
            chunk = corpus.get_chunk_by_id(cid)
            for w in set(re.findall(r"[a-z][a-z\-]{3,}", (chunk.text if chunk else "").lower())) - _STOP:
                terms[w] += 1
        group_terms[t] = terms
        doc_freq.update(set(terms))

    topics = []
    n_groups = len(groups)
    qv = _norm(np.asarray(query_vector)) if query_vector is not None else None
    for t, chunk_ids in groups.items():
        papers = {kg.chunk_meta[c]["paper_id"] for c in chunk_ids if c in kg.chunk_meta}
        member_vecs = [vectors[c] for c in chunk_ids if c in vectors]
        centroid = _norm(np.mean(member_vecs, axis=0)) if member_vecs else None
        ent = Counter()
        for cid in chunk_ids:
            for eid in kg.mentions.get(cid, {}):
                e = kg.entities.get(eid)
                if e and not e.get("generic") and not e.get("query_term"):
                    ent[eid] += 1
        tfidf = {w: c * math.log(1 + n_groups / doc_freq[w]) for w, c in group_terms[t].items()}
        terms = [w for w, _ in sorted(tfidf.items(), key=lambda kv: -kv[1])[:5]]
        top_entities = [e for e, _ in ent.most_common(4)]
        recent = sum(1 for p in papers if y_ref and (kg.papers.get(p, {}).get("year") or 0) >= y_ref - 1)
        topics.append({
            "id": t, "chunks": chunk_ids, "papers": sorted(papers), "size": len(chunk_ids),
            "centroid": centroid, "terms": terms, "entities": top_entities,
            "label": ", ".join([kg.entities[e]["name"] for e in top_entities[:3]] or terms[:3]),
            "recent_share": recent / len(papers) if papers else 0.0,
            "relevance": float(centroid @ qv) if centroid is not None and qv is not None else None,
        })
    multi = [len(t["papers"]) for t in topics if len(t["papers"]) >= 2] or [len(t["papers"]) for t in topics]
    median_papers = float(np.median(multi))
    large = [t for t in topics if len(t["papers"]) >= median_papers and t["centroid"] is not None]
    for t in topics:
        t["density"] = round(min(1.0, len(t["papers"]) / median_papers) if median_papers else 1.0, 4)
        others = [o for o in large if o["id"] != t["id"]]
        max_cos = max((float(t["centroid"] @ o["centroid"]) for o in others), default=0.0) if t["centroid"] is not None else 0.0
        semantic = min(1.0, max(0.0, (1 - max_cos) / 0.3))
        t["novelty"] = round(0.5 * semantic + 0.5 * t["recent_share"], 4)
        t["max_similarity_to_large_topics"] = round(max_cos, 4)
    chunk_topic: dict[str, int] = {}
    for t in sorted(topics, key=lambda t: len(t["papers"])):   # a chunk in several concept topics -> the sparsest
        for cid in t["chunks"]:
            chunk_topic.setdefault(cid, t["id"])
    return {"topics": topics, "chunk_topic": chunk_topic, "noise_chunks": noise,
            "median_papers": median_papers, "topic_method": method}


def _degenerate(kg: KnowledgeGraph, groups: dict[int, list[str]]) -> bool:
    n = max(1, len(kg.on_topic_papers))
    paper_counts = [len({kg.chunk_meta[c]["paper_id"] for c in g if c in kg.chunk_meta}) for g in groups.values()]
    multi = [c for c in paper_counts if c >= 2]
    return len(multi) < 3 or (max(paper_counts, default=0) / n > 0.6)


def _concept_groups(kg: KnowledgeGraph) -> dict[int, list[str]]:
    """Louvain concept clusters over specific concepts -> chunks mentioning >= 2 of a cluster's concepts."""
    import networkx as nx

    g = nx.Graph()
    specific = {e for e, v in kg.entities.items() if not v.get("generic") and not v.get("query_term") and len(v["papers"]) >= 2}
    for edge in kg.cooccurrence:
        if edge["a"] in specific and edge["b"] in specific and edge["papers"] >= 2 and edge["pmi"] > 0:
            g.add_edge(edge["a"], edge["b"], weight=edge["papers"] * edge["pmi"])
    if g.number_of_edges() == 0:
        return {}
    communities = nx.community.louvain_communities(g, weight="weight", resolution=CONCEPT_RESOLUTION, seed=42)
    groups: dict[int, list[str]] = {}
    for i, members in enumerate(sorted(communities, key=lambda m: -len(m))):
        if len(members) < 2:
            continue
        need = 2 if len(members) >= 3 else 1
        chunks = [cid for cid, counts in kg.mentions.items() if sum(1 for e in members if e in counts) >= need]
        if chunks:
            groups[i] = chunks
    return groups


def _years(kg: KnowledgeGraph) -> tuple[Optional[int], Optional[int]]:
    years = [kg.papers[p]["year"] for p in kg.on_topic_papers if kg.papers[p].get("year")]
    return (min(years), max(years)) if years else (None, None)


def topic_of(context: dict, chunk_ids: list[str]) -> Optional[dict]:
    """The topic most of these chunks belong to."""
    votes = Counter(context["chunk_topic"][c] for c in chunk_ids if c in context.get("chunk_topic", {}))
    if not votes:
        return None
    tid = votes.most_common(1)[0][0]
    return next((t for t in context["topics"] if t["id"] == tid), None)


def detect_sparse_topics(kg: KnowledgeGraph, context: dict) -> list[RawCandidate]:
    topics = context["topics"]
    if not topics:
        return []
    rels = sorted(t["relevance"] for t in topics if t["relevance"] is not None)
    min_rel = (rels[len(rels) // 2] - 0.05) if rels else None
    n = len(kg.on_topic_papers)
    out = []
    for t in topics:
        if t["density"] > SPARSE_DENSITY or len(t["papers"]) < 2:
            continue
        if min_rel is not None and t["relevance"] is not None and t["relevance"] < min_rel:
            continue
        statement = (f"Few papers address {t['label']} together: only {len(t['papers'])} of {n} papers "
                     f"(density {t['density']:.2f} of the median topic).")
        # evidence: chunks nearest the centroid, one per paper
        seen, evidence = set(), []
        for cid in t["chunks"]:
            pid = kg.chunk_meta[cid]["paper_id"]
            if pid not in seen:
                seen.add(pid)
                evidence.append((cid, f"in topic '{t['label']}'"))
        out.append(RawCandidate(
            detector=DETECTOR, statement=statement, core_entities=t["entities"][:3], evidence_chunks=evidence[:4],
            signals={"topic_id": t["id"], "topic_label": t["label"], "topic_terms": t["terms"],
                     "topic_papers": len(t["papers"]), "density": t["density"], "novelty": t["novelty"],
                     "relevance": round(t["relevance"], 4) if t["relevance"] is not None else None},
        ))
    return out
