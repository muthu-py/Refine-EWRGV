"""
app/knowledge_graph/builder.py
--------------------------------
Builds a run's KnowledgeGraph from its corpus (RunCorpus).

Steps
-----
1. Relevance gate: cosine(question+expansions centroid, title+abstract) per
   paper; papers below ``relevance_threshold`` are kept but marked
   off-topic and skipped by tagging and gap detection.
2. Noise exclusion: reference lists / acknowledgments / boilerplate chunks
   (app.retrieval.filters) are not tagged.
3. Entity tagging (EntityTagger: lexicons + query understanding + regex).
4. Canonicalisation: entity names of the same type whose embeddings have
   cosine >= ``unify_threshold`` (or equal normalised forms) are merged; the
   name with the most papers wins; all surface forms are kept as aliases.
5. Gap statements (statements.py) with their entities.
6. Paper-level co-occurrence (PMI) and Louvain communities.

Everything is deterministic for a fixed corpus and embedding model.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Optional

from app.core.logging import get_logger
from app.knowledge_graph.graph import (
    KnowledgeGraph,
    build_cooccurrence,
    entity_id,
    louvain_communities,
)
from app.knowledge_graph.claims import extract_result_claims
from app.knowledge_graph.lexicon import ENTITY_TYPES
from app.knowledge_graph.statements import find_gap_statements
from app.knowledge_graph.tagger import EntityTagger, normalise_term, query_terms
from app.retrieval.filters import is_noise

logger = get_logger(__name__)

# Relevance gate: a paper is off-topic when its similarity to the question is
# below max(floor, median_of_run - margin).  bge scores everything in a
# retrieved corpus high (0.6-0.9), so an absolute cut-off does not separate
# topics; the run-relative margin does (calibrated on run e0b54bcc: crop
# disease 0.750 and an ML-for-biologists guide 0.637 vs on-topic >= 0.764).
DEFAULT_RELEVANCE_FLOOR = 0.55
DEFAULT_RELEVANCE_MARGIN = 0.08
DEFAULT_UNIFY_THRESHOLD = 0.88

# Curated canonical names never merge with each other (e.g. sensitivity vs
# specificity embed close together but are different metrics).
_LEXICON_NAMES: dict[str, frozenset[str]] = {t: frozenset(entries) for t, entries in ENTITY_TYPES.items()}


def _norm(v: list[float]) -> list[float]:
    s = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / s for x in v]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _year(paper: Any) -> Optional[int]:
    if getattr(paper, "publication_year", None):
        return paper.publication_year
    date = getattr(paper, "publication_date", None)
    return date.year if date else None


class KnowledgeGraphBuilder:
    """
    Parameters
    ----------
    embedder:
        Object with ``embed_batch(texts) -> list[vector]`` (the run's
        embedding provider; the local bge model by default).
    relevance_floor / relevance_margin:
        Relevance gate: off-topic when cosine < max(floor, run median - margin).
    relevance_threshold:
        Fixed cut-off instead of the run-relative rule (0 disables the gate).
    unify_threshold:
        Cosine cut-off for synonym merging.
    """

    def __init__(
        self,
        embedder: Any,
        relevance_floor: float = DEFAULT_RELEVANCE_FLOOR,
        relevance_margin: float = DEFAULT_RELEVANCE_MARGIN,
        unify_threshold: float = DEFAULT_UNIFY_THRESHOLD,
        relevance_threshold: Optional[float] = None,
    ) -> None:
        self._embedder = embedder
        self._relevance_floor = relevance_floor
        self._relevance_margin = relevance_margin
        self._fixed_threshold = relevance_threshold  # overrides the run-relative rule
        self._unify_threshold = unify_threshold

    def build(
        self,
        corpus: Any,
        run_id: str,
        question: str,
        expanded_queries: Optional[list[str]] = None,
        understanding: Optional[dict] = None,
        query_id: Optional[str] = None,
    ) -> KnowledgeGraph:
        chunks = corpus.get_all_chunks()
        papers = {p.paper_id: p for p in corpus.get_all_papers()}
        chunk_papers = {c.paper_id for c in chunks}

        # ---- 1. relevance gate ---------------------------------------- #
        queries = [q for q in [question, *(expanded_queries or [])] if q and q.strip()]
        centroid = _norm([sum(col) / len(col) for col in zip(*[_norm(v) for v in self._embedder.embed_batch(queries)])])
        first_chunk: dict[str, str] = {}
        for c in chunks:
            first_chunk.setdefault(c.paper_id, c.text)
        paper_ids = sorted(chunk_papers)
        texts = []
        for pid in paper_ids:
            p = papers.get(pid)
            body = (p.abstract if p and p.abstract else first_chunk.get(pid, ""))[:1500]
            texts.append(f"{p.title if p else ''}. {body}")
        rel = [_dot(centroid, _norm(v)) for v in self._embedder.embed_batch(texts)] if texts else []
        if self._fixed_threshold is not None:
            threshold = self._fixed_threshold
        else:
            ordered = sorted(rel)
            median = ordered[len(ordered) // 2] if ordered else 0.0
            threshold = max(self._relevance_floor, median - self._relevance_margin)
        kg_papers: dict[str, dict] = {}
        for pid, score in zip(paper_ids, rel):
            p = papers.get(pid)
            kg_papers[pid] = {
                "title": p.title if p else pid,
                "year": _year(p) if p else None,
                "venue": getattr(p, "venue", None) if p else None,
                "doi": getattr(p, "doi", None) if p else None,
                "relevance": round(score, 4),
                "on_topic": score >= threshold,
            }
        on_topic = {pid for pid, p in kg_papers.items() if p["on_topic"]}

        # ---- 2-3. tag non-noise chunks of on-topic papers ------------- #
        tagger = EntityTagger(query_terms(understanding))
        raw_mentions: dict[str, dict[str, int]] = {}
        surfaces: dict[str, set[str]] = {}
        chunk_meta: dict[str, dict] = {}
        noise_chunks = 0
        tagged_chunks = []
        for c in chunks:
            if c.paper_id not in on_topic:
                continue
            stype = getattr(c.section_type, "value", str(c.section_type))
            if is_noise(stype, c.text):
                noise_chunks += 1
                continue
            chunk_meta[c.chunk_id] = {
                "paper_id": c.paper_id, "section": c.section_title, "section_type": stype,
                "page_start": c.page_start, "page_end": c.page_end,
            }
            counts: dict[str, int] = {}
            for m in tagger.tag(c.text):
                eid = entity_id(m.entity_type, m.canonical)
                counts[eid] = counts.get(eid, 0) + 1
                surfaces.setdefault(eid, set()).add(m.surface)
            raw_mentions[c.chunk_id] = counts
            tagged_chunks.append(c)

        # ---- 4. canonicalisation ------------------------------------- #
        raw_papers: dict[str, set[str]] = {}
        for cid, counts in raw_mentions.items():
            for eid in counts:
                raw_papers.setdefault(eid, set()).add(chunk_meta[cid]["paper_id"])
        mapping = self._unify(raw_papers)
        mentions: dict[str, dict[str, int]] = {}
        for cid, counts in raw_mentions.items():
            merged: dict[str, int] = {}
            for eid, n in counts.items():
                target = mapping.get(eid, eid)
                merged[target] = merged.get(target, 0) + n
            mentions[cid] = merged

        entities: dict[str, dict] = {}
        for cid, counts in mentions.items():
            pid = chunk_meta[cid]["paper_id"]
            for eid, n in counts.items():
                etype, name = eid.split(":", 1)
                e = entities.setdefault(eid, {"type": etype, "name": name, "aliases": set(),
                                              "papers": set(), "chunks": 0, "mentions": 0})
                e["papers"].add(pid)
                e["chunks"] += 1
                e["mentions"] += n
        for raw, target in list(mapping.items()) + [(e, e) for e in entities]:
            if target in entities:
                entities[target]["aliases"].update(surfaces.get(raw, set()))
        n_on = max(1, len(on_topic))
        query_text = " ".join(queries).lower()
        for e in entities.values():
            e["aliases"] = sorted(e["aliases"], key=str.lower)
            e["papers"] = sorted(e["papers"])
            # In >= 80% of papers: says nothing about a gap (e.g. "detection", "accuracy").
            e["generic"] = len(e["papers"]) / n_on >= 0.8 and n_on >= 5
            # Part of the question itself (e.g. "retinal disease"): the topic, not a gap dimension.
            e["query_term"] = normalise_term(e["name"]) in normalise_term(query_text) or any(
                normalise_term(a) in normalise_term(query_text) for a in e["aliases"] if len(a) > 3
            )

        # ---- 5. gap statements -------------------------------------- #
        statements: list[dict] = []
        result_claims: list[dict] = []
        seen_sentences: set[tuple[str, str]] = set()  # chunk overlap repeats sentences
        for c in tagged_chunks:
            meta = chunk_meta[c.chunk_id]
            for claim in extract_result_claims(c.text, meta["section_type"], tagger):
                result_claims.append({
                    "id": f"{c.chunk_id}:claim:{len(result_claims)}",
                    "paper_id": c.paper_id,
                    "chunk_id": c.chunk_id,
                    "section": c.section_title,
                    "section_type": meta["section_type"],
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "year": kg_papers.get(c.paper_id, {}).get("year"),
                    **{**claim, "entities": sorted({mapping.get(e, e) for e in claim["entities"]})},
                })
            for hit in find_gap_statements(c.text, meta["section_type"], c.section_title or ""):
                key = (c.paper_id, " ".join(hit.sentence.lower().split()))
                if key in seen_sentences:
                    continue
                seen_sentences.add(key)
                sent_entities = sorted({
                    mapping.get(entity_id(m.entity_type, m.canonical), entity_id(m.entity_type, m.canonical))
                    for m in tagger.tag(hit.sentence)
                })
                statements.append({
                    "id": f"{c.chunk_id}:{len(statements)}",
                    "paper_id": c.paper_id,
                    "chunk_id": c.chunk_id,
                    "section": c.section_title,
                    "section_type": meta["section_type"],
                    "page_start": c.page_start,
                    "page_end": c.page_end,
                    "year": kg_papers.get(c.paper_id, {}).get("year"),
                    "sentence": hit.sentence,
                    "cue_class": hit.cue_class,
                    "cue": hit.cue,
                    "strength": hit.strength,
                    "entities": [e for e in sent_entities if e in entities],
                })

        # ---- 6. co-occurrence + communities ------------------------- #
        cooccurrence = build_cooccurrence(entities, len(on_topic))
        communities = louvain_communities(entities, cooccurrence)
        for c in communities:
            for eid in c["entities"]:
                entities[eid]["community"] = c["id"]

        by_type: dict[str, int] = {}
        for e in entities.values():
            by_type[e["type"]] = by_type.get(e["type"], 0) + 1
        by_cue: dict[str, int] = {}
        for s in statements:
            by_cue[s["cue_class"]] = by_cue.get(s["cue_class"], 0) + 1
        stats = {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "papers": len(kg_papers),
            "on_topic_papers": len(on_topic),
            "off_topic_papers": len(kg_papers) - len(on_topic),
            "chunks_total": len(chunks),
            "chunks_tagged": len(tagged_chunks),
            "chunks_noise_skipped": noise_chunks,
            "entities": len(entities),
            "entities_by_type": dict(sorted(by_type.items())),
            "entities_merged_as_synonyms": sum(1 for k, v in mapping.items() if k != v),
            "mentions": sum(sum(c.values()) for c in mentions.values()),
            "cooccurrence_edges": len(cooccurrence),
            "communities": len(communities),
            "gap_statements": len(statements),
            "result_claims": len(result_claims),
            "gap_statements_by_cue": by_cue,
            "vocabulary_size": tagger.vocabulary_size,
        }
        logger.info("KnowledgeGraphBuilder: built", extra={"run_id": run_id, **{k: v for k, v in stats.items() if isinstance(v, int)}})
        return KnowledgeGraph(
            run_id=run_id,
            query_id=query_id,
            papers=kg_papers,
            entities=entities,
            mentions=mentions,
            chunk_meta=chunk_meta,
            cooccurrence=cooccurrence,
            statements=statements,
            result_claims=result_claims,
            communities=communities,
            stats=stats,
            params={
                "question": question,
                "expanded_queries": list(expanded_queries or []),
                "relevance_threshold": round(threshold, 4),
                "unify_threshold": self._unify_threshold,
                "embedding_model": getattr(self._embedder, "get_model_name", lambda: "unknown")(),
            },
        )

    # ------------------------------------------------------------------ #

    def _unify(self, entity_papers: dict[str, set[str]]) -> dict[str, str]:
        """Map raw entity id -> canonical entity id (synonym merging within a type)."""
        by_type: dict[str, list[str]] = {}
        for eid in entity_papers:
            by_type.setdefault(eid.split(":", 1)[0], []).append(eid)
        mapping: dict[str, str] = {}
        for etype, eids in by_type.items():
            # strongest (most papers) first, so it becomes the canonical name
            eids.sort(key=lambda e: (-len(entity_papers[e]), e))
            names = [e.split(":", 1)[1] for e in eids]
            vectors = [_norm(v) for v in self._embedder.embed_batch(names)] if names else []
            canon: list[int] = []
            for i, eid in enumerate(eids):
                target = None
                for j in canon:
                    if names[i] in _LEXICON_NAMES.get(etype, ()) and names[j] in _LEXICON_NAMES.get(etype, ()):
                        continue  # two curated entities are distinct by definition
                    if normalise_term(names[i]) == normalise_term(names[j]) or _dot(vectors[i], vectors[j]) >= self._unify_threshold:
                        target = eids[j]
                        break
                if target is None:
                    canon.append(i)
                    mapping[eid] = eid
                else:
                    mapping[eid] = target
        return mapping
