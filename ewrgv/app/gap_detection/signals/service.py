"""
app/gap_detection/signals/service.py
--------------------------------------
End-to-end signal-based gap detection for one run.

    KG + corpus -> D1, D2 -> fuse -> GapScore v1 -> grounding gate
                -> (optional LLM wording, gated) -> rank -> readable candidates

The LLM never decides that a gap exists and its confidence is never scored.
With ``llm_wording`` it only rewrites candidate wording based on its existing
evidence; the rewrite is kept only if it passes the grounding gate. With
``include_llm_baseline`` the existing LLM detector (app/gap_detection/
detector.py) is run on the same corpus and stored separately for comparison.
"""

from __future__ import annotations

import math
from typing import Any, Optional
from uuid import uuid4

from app.core.config import settings
from app.core.logging import get_logger
from app.gap_detection.signals.combination import DETECTOR as D2, detect_combinations
from app.gap_detection.signals.explicit import DETECTOR as D1, RawCandidate, detect_explicit
from app.gap_detection.signals.fusion import fuse, gap_type_for, rank, representation, score
from app.gap_detection.signals.temporal import DETECTOR as D3, detect_stale, staleness_context
from app.gap_detection.signals.topics import DETECTOR as D4, build_topics, detect_sparse_topics
from app.knowledge_graph.graph import KnowledgeGraph

logger = get_logger(__name__)

GROUNDING_THRESHOLD = 0.60
GROUNDING_ATTEMPTS = 3
ALL_DETECTORS = (D1, D2, D3, D4)
DETECTOR_ALIASES = {"explicit": D1, "d1": D1, D1: D1, "combination": D2, "d2": D2, D2: D2,
                    "temporal": D3, "d3": D3, D3: D3, "topics": D4, "d4": D4, D4: D4}
TREND_CONCEPTS = 25   # specific concepts (most papers first) whose external trend is fetched


def _norm(v) -> list[float]:
    s = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / s for x in v]


def _cos(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


class GapDetectionService:
    def __init__(self, kg: KnowledgeGraph, corpus: Any, embedder: Any, retrieval: Any = None,
                 llm: Any = None, domain: Optional[str] = None, external: Any = None,
                 queries: Optional[list[str]] = None) -> None:
        self._kg = kg
        self._corpus = corpus
        self._embedder = embedder
        self._retrieval = retrieval
        self._llm = llm
        self._domain = domain
        self._external = external          # ExternalLiterature for OpenAlex trends (None = internal only)
        self._queries = queries or [kg.params.get("question") or ""] + list(kg.params.get("expanded_queries") or [])
        self._context: dict = {}

    # ------------------------------------------------------------------ #

    async def run(self, detectors: Optional[list[str]] = None, llm_wording: bool = False,
                  progress=None) -> dict:
        import asyncio

        if self._retrieval is not None:
            await _report(progress, "retrieving and reranking gap evidence")
            from app.retrieval.filters import RetrievalFilters

            retrieval_queries = list(dict.fromkeys(
                q.strip() for q in self._queries if q and q.strip()
            ))
            self._context["reranked_query_results"] = await self._retrieval.aretrieve_many(
                retrieval_queries,
                top_k=min(100, settings.RERANK_MAX_CANDIDATES),
                method="hybrid",
                filters=RetrievalFilters.build(section_types=["gap_evidence"]),
            ) if retrieval_queries else []

        chosen = [DETECTOR_ALIASES[d] for d in (detectors or ALL_DETECTORS) if d in DETECTOR_ALIASES]
        chosen = list(dict.fromkeys(chosen)) or list(ALL_DETECTORS)
        raw: list[RawCandidate] = []
        per_detector: dict[str, int] = {}
        if D1 in chosen:
            await _report(progress, "D1 explicit statements")
            d1 = await asyncio.to_thread(detect_explicit, self._kg, self._embedder)
            raw += d1
            per_detector[D1] = len(d1)
        if D2 in chosen:
            await _report(progress, "D2 combination sparsity")
            d2 = detect_combinations(self._kg)
            raw += d2
            per_detector[D2] = len(d2)

        await _report(progress, "D4 topic clusters")
        texts = [q for q in self._queries if q and q.strip()]
        qvec = None
        if texts:
            vecs = [_norm(v) for v in self._embedder.embed_batch(texts)]
            qvec = _norm([sum(col) / len(col) for col in zip(*vecs)])
        topics = await asyncio.to_thread(build_topics, self._kg, self._corpus, qvec)
        self._context["topics"] = topics
        if D4 in chosen:
            d4 = detect_sparse_topics(self._kg, topics)
            raw += d4
            per_detector[D4] = len(d4)

        await _report(progress, "D3 temporal staleness" + (" (OpenAlex trends)" if self._external else ""))
        specific = sorted(
            (e for e, v in self._kg.entities.items() if not v.get("generic") and not v.get("query_term")),
            key=lambda e: -len(self._kg.entities[e]["papers"]))
        concepts = list(dict.fromkeys([e for c in raw for e in c.core_entities] + specific))
        # external trends only for the most frequent concepts and the candidates' concepts
        ordered = list(dict.fromkeys([e for c in raw for e in c.core_entities][:TREND_CONCEPTS] + specific[:TREND_CONCEPTS]))
        staleness = await staleness_context(self._kg, ordered + [c for c in concepts if c not in ordered], self._external)
        self._context["staleness"] = staleness
        if D3 in chosen:
            d3 = detect_stale(self._kg, staleness)
            raw += d3
            per_detector[D3] = len(d3)

        await _report(progress, "fusion and scoring")
        groups = fuse(self._kg, raw)
        candidates = [self._candidate(g) for g in groups]

        await _report(progress, "grounding gate")
        for c in candidates:
            await self._ground(c)
        if llm_wording and self._llm is not None:
            await _report(progress, "LLM wording")
            for c in candidates:
                await self._reword(c)

        ranked = rank(candidates)
        reranked_results = self._context.get("reranked_query_results", [])
        reranking = (
            (reranked_results[0].metadata or {}).get("reranking", {})
            if reranked_results else {"status": "no_candidates", "candidates_scored": 0}
        )
        stats = {
            "candidates_per_detector": per_detector,
            "candidates_after_fusion": len(ranked),
            "merged_multi_detector": sum(1 for c in ranked if len(c["detectors"]) > 1),
            "grounded": sum(1 for c in ranked if c["grounding"]["grounded"]),
            "uncertain": sum(1 for c in ranked if c["status"] == "uncertain"),
            "by_gap_type": _count(c["gap_type"] for c in ranked),
            "multi_paper": sum(1 for c in ranked if c["score_notes"]["stating_papers"] >= 2),
            "topics": len(topics["topics"]),
            "topic_method": topics.get("topic_method"),
            "topic_noise_chunks": topics.get("noise_chunks"),
            "trend_anchor": staleness.get("anchor"),
            "corpus_span_years": staleness.get("corpus_span"),
            "concepts_with_external_trend": sum(1 for v in staleness["by_concept"].values() if v.get("tau_external") is not None),
            "reranked_context_candidates": len(reranked_results),
            "reranking": reranking,
        }
        topic_summary = [
            {"id": t["id"], "label": t["label"], "papers": len(t["papers"]), "chunks": t["size"],
             "density": t["density"], "novelty": t["novelty"], "terms": t["terms"],
             "relevance": round(t["relevance"], 4) if t["relevance"] is not None else None}
            for t in sorted(topics["topics"], key=lambda t: t["density"])
        ]
        trends = {
            self._kg.entities[e]["name"]: {k: v for k, v in info.items() if k in (
                "tau", "tau_internal", "tau_external", "last_year", "external_rel_slope", "external_share")}
            for e, info in staleness["by_concept"].items() if info.get("tau_external") is not None
        }
        return {"stats": stats, "candidates": ranked, "topics": topic_summary, "trends": trends}

    # ------------------------------------------------------------------ #

    def _candidate(self, group: dict) -> dict:
        kg = self._kg
        primary: RawCandidate = group["primary"]
        scored = score(kg, group, self._context)
        core = primary.core_entities
        statement = primary.statement
        evidence = list(dict.fromkeys(primary.evidence_chunks + [e for x in group["extra"] for e in x.evidence_chunks]))
        counters = list(dict.fromkeys(primary.counter_chunks + [e for x in group["extra"] for e in x.counter_chunks]))
        signals = {primary.detector: primary.signals}
        for x in group["extra"]:
            signals[x.detector] = x.signals
        gtype = gap_type_for(kg, core, statement, group["detectors"])
        return {
            "gap_id": str(uuid4()),
            "rank": None,
            "gap_statement": statement,
            "original_statement": statement,
            "gap_type": gtype.value,
            "status": "candidate",
            "candidate_source": "signal",
            "detectors": list(dict.fromkeys(group["detectors"])),
            **scored,
            "concepts": [
                {"id": e, "name": kg.entities[e]["name"], "type": kg.entities[e]["type"],
                 "papers": len(kg.entities[e]["papers"])}
                for e in core if e in kg.entities
            ],
            "signals": signals,
            "supporting_statements": [self._statement_view(s) for s in primary.statements[:8]],
            "evidence_passages": [self._passage(cid, role) for cid, role in evidence[:10]],
            "counter_hints": [self._passage(cid, reason) for cid, reason in counters[:5]],
            "grounding": {"grounded": False, "max_similarity": None, "method": None, "attempts": 0},
            "structured_representation": representation(kg, core, statement, self._domain),
        }

    def _paper(self, pid: str) -> dict:
        p = self._kg.papers.get(pid, {})
        return {"paper_id": pid, "title": p.get("title"), "year": p.get("year"), "doi": p.get("doi")}

    def _statement_view(self, s: dict) -> dict:
        return {
            **self._paper(s["paper_id"]),
            "chunk_id": s["chunk_id"], "section": s["section"], "section_type": s["section_type"],
            "page_start": s["page_start"], "page_end": s["page_end"],
            "sentence": s["sentence"], "cue_class": s["cue_class"], "cue": s["cue"], "strength": s["strength"],
        }

    def _passage(self, chunk_id: str, role: str) -> dict:
        chunk = self._corpus.get_chunk_by_id(chunk_id) if self._corpus is not None else None
        meta = self._kg.chunk_meta.get(chunk_id, {})
        pid = meta.get("paper_id") or (chunk.paper_id if chunk else None)
        text = " ".join((chunk.text if chunk else "").split())
        return {
            **self._paper(pid),
            "chunk_id": chunk_id, "role": role,
            "section": meta.get("section") or (chunk.section_title if chunk else None),
            "section_type": meta.get("section_type"),
            "page_start": meta.get("page_start"), "page_end": meta.get("page_end"),
            "snippet": text[:400] + ("..." if len(text) > 400 else ""),
        }

    # ------------------------------------------------------------------ #
    # Grounding gate (ResearchRAG CSCG idea): the statement must align with evidence.
    # ------------------------------------------------------------------ #

    def _evidence_vectors(self, c: dict) -> list[list[float]]:
        vectors = []
        sentences = [s["sentence"] for s in c["supporting_statements"]]
        if sentences:
            vectors += [_norm(v) for v in self._embedder.embed_batch(sentences)]
        for p in c["evidence_passages"]:
            vec = self._corpus.get_embedding(p["chunk_id"]) if hasattr(self._corpus, "get_embedding") else None
            if vec:
                vectors.append(_norm(vec))
        return vectors

    async def _ground(self, c: dict) -> None:
        structural_only = c["detectors"][0] in (D2, D3, D4) and not c["supporting_statements"]
        if structural_only:
            # A template statement about counts / trends / topics: grounded when its
            # evidence exists (D2: both concepts have evidence passages).
            roles = {p["role"] for p in c["evidence_passages"]}
            needed = 2 if c["detectors"][0] == D2 else 1
            ok = sum(1 for r in roles if r.startswith(("mentions ", "in topic"))) >= needed
            c["grounding"] = {"grounded": ok, "max_similarity": None, "method": "structural", "attempts": 0}
            if not ok:
                c["status"] = "uncertain"
            return
        statement_vec = _norm(self._embedder.embed_batch([c["gap_statement"]])[0])
        best = max((_cos(statement_vec, v) for v in self._evidence_vectors(c)), default=0.0)
        attempts = 0
        known_evidence = {p["chunk_id"] for p in c["evidence_passages"]}
        for result in self._context.get("reranked_query_results", []):
            if result.chunk_id in known_evidence:
                continue
            vec = self._corpus.get_embedding(result.chunk_id)
            similarity = _cos(statement_vec, _norm(vec)) if vec else 0.0
            if similarity >= GROUNDING_THRESHOLD:
                c["evidence_passages"].append(
                    self._passage(result.chunk_id, "reranked query context")
                )
                known_evidence.add(result.chunk_id)
                best = max(best, similarity)
        if best < GROUNDING_THRESHOLD and self._retrieval is not None:
            from app.retrieval.filters import RetrievalFilters

            names = [x["name"] for x in c["concepts"]]
            queries = [c["gap_statement"], " ".join(names), " ".join(names + ["limitation"])]
            for q in queries[:GROUNDING_ATTEMPTS]:
                if not q.strip():
                    continue
                attempts += 1
                results = await self._retrieval.aretrieve_many(
                    [q], top_k=5, method="hybrid", filters=RetrievalFilters.build(section_types=["gap_evidence"]))
                for r in results:
                    vec = self._corpus.get_embedding(r.chunk_id)
                    if vec and _cos(statement_vec, _norm(vec)) >= GROUNDING_THRESHOLD:
                        c["evidence_passages"].append(self._passage(r.chunk_id, "re-retrieved (grounding)"))
                        best = max(best, _cos(statement_vec, _norm(vec)))
                if best >= GROUNDING_THRESHOLD:
                    break
        grounded = best >= GROUNDING_THRESHOLD
        c["grounding"] = {"grounded": grounded, "max_similarity": round(best, 4), "method": "semantic", "attempts": attempts}
        if not grounded:
            c["status"] = "uncertain"

    async def _reword(self, c: dict) -> None:
        evidence = [s["sentence"] for s in c["supporting_statements"] if s.get("sentence")]
        if not evidence:
            evidence = [p["snippet"] for p in c["evidence_passages"]
                        if p.get("snippet") and not str(p.get("role", "")).lower().startswith(("counter", "contradict"))]
        if not evidence:
            return
        from app.gap_detection.llm_call import call_structured_llm
        import asyncio

        original = c.get("original_statement") or c["gap_statement"]
        quotes = "\n".join(f"- {text}" for text in evidence[:6])
        prompt = (
            "Rewrite this research-gap candidate as one clear, specific sentence for a research interface. "
            "This is wording only: preserve its scope and uncertainty, do not strengthen it into a novelty claim, "
            "and do not add methods, datasets, populations, or claims that are not already supported.\n\n"
            f"Original candidate: {original}\n\n"
            f"Supporting passages:\n{quotes}\n\n"
            "Return JSON {\"gap_statement\": \"...\"}."
        )
        schema = {"type": "object", "properties": {"gap_statement": {"type": "string"}}, "required": ["gap_statement"]}
        try:
            out = await asyncio.to_thread(call_structured_llm, self._llm, prompt, schema, operation="gap wording")
            text = (out or {}).get("gap_statement", "").strip()
        except Exception as exc:  # noqa: BLE001 - wording is optional
            logger.warning("LLM wording failed: %s", exc)
            return
        if not text:
            return
        vec = _norm(self._embedder.embed_batch([text])[0])
        best = max((_cos(vec, v) for v in self._evidence_vectors(c)), default=0.0)
        if best >= GROUNDING_THRESHOLD:
            c["gap_statement"] = text
            c["wording"] = {
                "source": "llm", "original_statement": original,
                "grounding_similarity": round(best, 4),
            }
        else:
            c["wording"] = {
                "source": "original", "original_statement": original,
                "rejected_llm_wording": text, "grounding_similarity": round(best, 4),
            }


async def _report(progress, step: str) -> None:
    if progress is not None:
        await progress(step)


def _count(items) -> dict:
    out: dict = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return out
