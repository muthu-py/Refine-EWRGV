"""
app/knowledge_graph/jobs.py
-----------------------------
Background jobs behind the API:

    POST /research/kg/build  -> run_kg_build       (build + store the run's KG)
    POST /research/gaps      -> run_gap_detection  (KG -> D1/D2 -> GapScore -> store)

plus the read-side view builders used by GET /research/kg/{run_id} and
GET /research/gaps/{run_id}.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

from app.core.job_store import AnalysisJob, JobStatus
from app.core.logging import get_logger
from app.knowledge_graph.graph import KnowledgeGraph

logger = get_logger(__name__)


# ---------------------------------------------------------------------- #
# Shared loading
# ---------------------------------------------------------------------- #


async def resolve_query(run_id: str, query_id: Optional[str]) -> dict:
    """Question, expansions and understanding for a run (explicit query_id > the run's linked query)."""
    from app.storage.relational.database import get_db_pool
    from app.storage.repositories.query_repository import QueryRunRepository

    pool = await get_db_pool()
    row = await pool.fetchrow(
        "SELECT research_question, query_id::text AS query_id FROM research_runs WHERE run_id::text = $1", run_id)
    if row is None:
        raise LookupError(f"run_id '{run_id}' not found.")
    qid = query_id or row["query_id"]
    stored = await QueryRunRepository(pool).get(qid) if qid else None
    if query_id and stored is None:
        raise LookupError(f"query_id '{query_id}' not found.")
    return {
        "query_id": qid if stored else None,
        "question": (stored or {}).get("research_question") or row["research_question"],
        "expanded_queries": list((stored or {}).get("expanded_queries") or []),
        "understanding": (stored or {}).get("understanding"),
    }


async def load_corpus(run_id: str, embedder: Any) -> Any:
    """The run corpus with stored vectors (cached; see app.retrieval.corpus_cache)."""
    from app.ingestion.corpus.run_corpus import RunCorpusLoader
    from app.retrieval import corpus_cache
    from app.storage.relational.database import get_db_pool
    from app.storage.repositories.embedding_repository import EmbeddingRepository
    from app.storage.repositories.paper_repository import PaperRepository
    from app.storage.supabase_storage import SupabaseStorageClient

    pool = await get_db_pool()
    loader = RunCorpusLoader(paper_repo=PaperRepository(pool), storage_client=SupabaseStorageClient(),
                             embedding_repo=EmbeddingRepository(pool))
    corpus = await corpus_cache.get_or_load(run_id, embedder.get_model_name(), lambda: loader.load(run_id))
    if not corpus.get_all_chunks():
        raise ValueError(f"run_id '{run_id}' has no chunks; run /chunks and /represent first.")
    return corpus


async def _set(job: AnalysisJob, store: Any, **changes) -> None:
    for k, v in changes.items():
        setattr(job, k, v)
    await store.update_analysis_job(job)


# ---------------------------------------------------------------------- #
# KG build
# ---------------------------------------------------------------------- #


async def build_and_store_kg(run_id: str, query_id: Optional[str], embedder: Any,
                             relevance_threshold: Optional[float] = None) -> KnowledgeGraph:
    from app.knowledge_graph.builder import KnowledgeGraphBuilder
    from app.storage.relational.database import get_db_pool
    from app.storage.repositories.kg_repository import KnowledgeGraphRepository

    query = await resolve_query(run_id, query_id)
    corpus = await load_corpus(run_id, embedder)
    builder = KnowledgeGraphBuilder(embedder, relevance_threshold=relevance_threshold)
    kg = await asyncio.to_thread(builder.build, corpus, run_id, query["question"], query["expanded_queries"],
                                 query["understanding"], query["query_id"])
    await KnowledgeGraphRepository(await get_db_pool()).save_graph(run_id, kg.query_id, kg.stats, kg.to_payload())
    return kg


async def run_kg_build(job: AnalysisJob, store: Any, query_id: Optional[str] = None,
                       relevance_threshold: Optional[float] = None) -> None:
    from app.providers.embeddings.factory import create_embedding_provider

    await _set(job, store, status=JobStatus.RUNNING, started_at=datetime.now(timezone.utc), progress="loading corpus")
    try:
        embedder = create_embedding_provider()
        await _set(job, store, progress="building knowledge graph")
        kg = await build_and_store_kg(job.run_id, query_id, embedder, relevance_threshold)
        await _set(job, store, status=JobStatus.COMPLETED, progress="done", summary=kg.stats)
    except Exception as exc:  # noqa: BLE001 - reported on the job
        logger.error("KG build failed for run %s: %s", job.run_id, exc, exc_info=True)
        await _set(job, store, status=JobStatus.FAILED, progress="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        job.completed_at = datetime.now(timezone.utc)
        await store.update_analysis_job(job)


# ---------------------------------------------------------------------- #
# Gap detection
# ---------------------------------------------------------------------- #


async def run_gap_detection(job: AnalysisJob, store: Any, query_id: Optional[str] = None,
                            detectors: Optional[list[str]] = None, llm_wording: bool = False,
                            include_llm_baseline: bool = False, rebuild_kg: bool = False) -> None:
    from app.gap_detection.signals.service import GapDetectionService
    from app.providers.embeddings.factory import create_embedding_provider
    from app.retrieval.service import RetrievalService
    from app.storage.relational.database import get_db_pool
    from app.storage.repositories.kg_repository import KnowledgeGraphRepository
    from app.validation.sources import ExternalLiterature

    await _set(job, store, status=JobStatus.RUNNING, started_at=datetime.now(timezone.utc), progress="loading")
    try:
        embedder = create_embedding_provider()
        repo = KnowledgeGraphRepository(await get_db_pool())
        stored = None if rebuild_kg else await repo.get_graph(job.run_id)
        if stored is not None and query_id and stored["query_id"] not in (None, query_id):
            stored = None  # built for a different query
        if stored is None:
            await _set(job, store, progress="building knowledge graph")
            kg = await build_and_store_kg(job.run_id, query_id, embedder)
        else:
            kg = KnowledgeGraph.from_payload(stored["payload"])
        corpus = await load_corpus(job.run_id, embedder)
        query = await resolve_query(job.run_id, query_id or kg.query_id)

        llm = None
        if llm_wording or include_llm_baseline:
            from app.core.config import settings
            from app.providers.llm.factory import create_llm_provider

            if settings.LLM_API_KEY or settings.OPENROUTER_API_KEY:
                settings.LLM_TIMEOUT = max(settings.LLM_TIMEOUT, 180)
                llm = create_llm_provider()

        async def progress(step: str) -> None:
            await _set(job, store, progress=step)

        domain = (query.get("understanding") or {}).get("domain")
        external = ExternalLiterature()
        try:
            service = GapDetectionService(kg, corpus, embedder, retrieval=RetrievalService(corpus, embedder),
                                          llm=llm, domain=domain, external=external)
            result = await service.run(detectors=detectors, llm_wording=llm_wording, progress=progress)
        finally:
            await external.close()

        baseline = None
        if include_llm_baseline:
            await _set(job, store, progress="LLM baseline")
            baseline = await asyncio.to_thread(_llm_baseline, corpus, query, embedder, llm)

        params = {"query_id": query.get("query_id"), "detectors": detectors or [
                      "explicit_statements", "combination_sparsity", "temporal_staleness", "topic_density"],
                  "llm_wording": llm_wording and llm is not None, "include_llm_baseline": include_llm_baseline,
                  "kg_built_at": kg.stats.get("built_at")}
        await repo.save_gaps(job.run_id, query.get("query_id"), params, result["stats"], result["candidates"], baseline)
        summary = dict(result["stats"])
        if baseline is not None:
            summary["llm_baseline_candidates"] = len(baseline.get("candidates", []))
        await _set(job, store, status=JobStatus.COMPLETED, progress="done", summary=summary)
    except Exception as exc:  # noqa: BLE001 - reported on the job
        logger.error("Gap detection failed for run %s: %s", job.run_id, exc, exc_info=True)
        await _set(job, store, status=JobStatus.FAILED, progress="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        job.completed_at = datetime.now(timezone.utc)
        await store.update_analysis_job(job)


def _llm_baseline(corpus: Any, query: dict, embedder: Any, llm: Any) -> dict:
    """The existing LLM-only detector (Baseline 1) on the same corpus, hybrid retrieval."""
    if llm is None:
        return {"error": "No LLM key configured; baseline not run.", "candidates": []}
    from app.core.config import settings
    from app.domain.models.query import ResearchQuery
    from app.domain.models.query_understanding import QueryUnderstanding
    from app.gap_detection.detector import GapDetector
    from app.gap_detection.evidence_scoring import CandidateEvidenceScorer
    from app.gap_detection.ranking import CandidateRanker
    from app.orchestration.research_pipeline import ResearchPipeline
    from app.retrieval.service import RetrievalService

    rq = ResearchQuery(
        query=query["question"], expanded_queries=query["expanded_queries"],
        understanding=QueryUnderstanding.model_validate(query["understanding"]) if query.get("understanding") else None,
    )
    pipeline = ResearchPipeline(
        None, None, None, None, None,
        gap_detector=GapDetector(llm, max_candidates=settings.MAX_CANDIDATE_GAPS),
        candidate_ranker=CandidateRanker(), ewrgv_validator=None, top_k_gaps=10, retrieval_top_k=20,
        retriever_factory=lambda c: RetrievalService(c, embedding_provider=embedder),
        evidence_scorer=CandidateEvidenceScorer(),
    )
    result = pipeline.run_on_corpus(rq, corpus)
    return {
        "detector": "llm (app/gap_detection/detector.py)",
        "model": f"{settings.LLM_PROVIDER}/{settings.LLM_MODEL}",
        "error": result.error,
        "chunks_sent": len(result.retrieved_chunks),
        "candidates": [
            {
                "rank": g.rank, "gap_statement": g.gap_statement, "gap_type": g.primary_gap_type.value,
                "llm_confidence": g.initial_confidence,
                "evidence_score": g.evidence_assessment.score if g.evidence_assessment else None,
                "rank_score": (g.metadata.get("ranking") or {}).get("adjusted_score"),
                "source_papers": len(g.source_papers), "supporting_passages": len(g.relevant_passages),
                "contradicting_passages": len(g.contradicting_passages),
            }
            for g in result.top_k_gaps
        ],
    }


# ---------------------------------------------------------------------- #
# Read-side views
# ---------------------------------------------------------------------- #


def kg_view(stored: dict, view: str = "summary", entity_type: Optional[str] = None, limit: int = 25) -> dict:
    kg = KnowledgeGraph.from_payload(stored["payload"])
    header = {"run_id": kg.run_id, "query_id": kg.query_id, "built_at": stored.get("built_at")}
    entity_type = entity_type.upper() if entity_type else None

    def entities(n: int) -> list[dict]:
        rows = [
            {"id": eid, "name": e["name"], "type": e["type"], "papers": len(e["papers"]), "mentions": e["mentions"],
             "community": e.get("community"), "generic": e.get("generic", False),
             "query_term": e.get("query_term", False), "aliases": e["aliases"][:8]}
            for eid, e in kg.entities.items() if entity_type is None or e["type"] == entity_type
        ]
        return sorted(rows, key=lambda r: (-r["papers"], r["name"]))[:n]

    if view == "entities":
        return {**header, "entities": entities(limit)}
    if view == "statements":
        rows = sorted(kg.statements, key=lambda s: -s["strength"])[:limit]
        return {**header, "statements": [
            {**{k: s[k] for k in ("sentence", "cue_class", "cue", "strength", "section", "section_type",
                                  "page_start", "page_end", "year")},
             "paper": kg.papers.get(s["paper_id"], {}).get("title"),
             "entities": [kg.entities[e]["name"] for e in s["entities"] if e in kg.entities]}
            for s in rows]}
    if view == "claims":
        rows = kg.result_claims[:limit]
        return {**header, "result_claims": [
            {**{k: claim.get(k) for k in ("sentence", "direction", "metric", "section", "section_type",
                                           "page_start", "page_end", "year")},
             "paper": kg.papers.get(claim["paper_id"], {}).get("title"),
             "entities": [kg.entities[e]["name"] for e in claim["entities"] if e in kg.entities]}
            for claim in rows]}
    if view == "communities":
        return {**header, "communities": [
            {**c, "entities": [kg.entities[e]["name"] for e in c["entities"][:limit] if e in kg.entities]}
            for c in kg.communities]}
    if view == "papers":
        rows = sorted(kg.papers.items(), key=lambda x: -x[1]["relevance"])
        return {**header, "relevance_threshold": kg.params.get("relevance_threshold"),
                "papers": [{"paper_id": pid, **p} for pid, p in rows]}
    if view == "cytoscape":
        return {**header, **kg.to_cytoscape()}
    # summary
    edges = sorted(kg.cooccurrence, key=lambda e: (-e["papers"], -e["pmi"]))
    strong = [
        {"a": kg.entities[e["a"]]["name"], "b": kg.entities[e["b"]]["name"], "papers": e["papers"], "pmi": e["pmi"]}
        for e in edges if not kg.entities[e["a"]].get("generic") and not kg.entities[e["b"]].get("generic")
    ][:10]
    return {
        **header,
        "stats": kg.stats,
        "off_topic_papers": [{"paper_id": pid, "title": p["title"], "relevance": p["relevance"]}
                             for pid, p in kg.papers.items() if not p["on_topic"]],
        "top_entities": entities(15),
        "communities": [{"id": c["id"], "label": c["label"], "size": c["size"]} for c in kg.communities],
        "strongest_cooccurrences": strong,
        "views": ["summary", "entities", "statements", "claims", "communities", "papers", "cytoscape", "graphml"],
    }


def gaps_view(stored: dict, view: str = "summary", top: int = 10, detector: Optional[str] = None) -> dict:
    candidates = stored["candidates"]
    if detector:
        candidates = [c for c in candidates if detector in c["detectors"]]
    header = {"run_id": stored["run_id"], "query_id": stored.get("query_id"), "created_at": stored.get("created_at"),
              "params": stored.get("params"), "stats": stored.get("stats")}
    if view == "full":
        return {**header, "candidates": candidates[:top], "llm_baseline": stored.get("baseline")}
    if view == "compare":
        return {
            **header,
            "signal_based_top": [_brief(c) for c in candidates[:top]],
            "llm_baseline_top": (stored.get("baseline") or {}).get("candidates", [])[:top],
            "note": None if stored.get("baseline") else "Run POST /research/gaps with include_llm_baseline=true to compare.",
        }
    return {**header, "top_gaps": [_card(c) for c in candidates[:top]]}


def _brief(c: dict) -> dict:
    return {"rank": c["rank"], "gap_statement": c["gap_statement"], "gap_type": c["gap_type"],
            "gap_score": c["gap_score"], "detectors": c["detectors"],
            "stating_papers": c["score_notes"]["stating_papers"], "evidence_passages": len(c["evidence_passages"])}


def _card(c: dict) -> dict:
    """A readable gap card: score breakdown, agreeing detectors, evidence with section and page."""
    return {
        "rank": c["rank"],
        "gap_id": c["gap_id"],
        "gap_statement": c["gap_statement"],
        "gap_type": c["gap_type"],
        "status": c["status"],
        "gap_score": c["gap_score"],
        "rank_score": c.get("rank_score"),
        "score_breakdown": c["score_breakdown"],
        "detectors_agreeing": c["detectors"],
        "concepts": [f"{x['name']} ({x['type']}, {x['papers']} papers)" for x in c["concepts"]],
        "stating_papers": c["score_notes"]["stating_papers"],
        "supporting_statements": [
            {"paper": s["title"], "year": s["year"], "section": s["section"], "section_type": s["section_type"],
             "pages": _pages(s), "cue": f"{s['cue_class']}: {s['cue']}", "quote": s["sentence"]}
            for s in c["supporting_statements"][:5]
        ],
        "evidence_passages": [
            {"paper": p["title"], "year": p["year"], "section": p["section"], "section_type": p["section_type"],
             "pages": _pages(p), "role": p["role"], "snippet": p["snippet"][:240]}
            for p in c["evidence_passages"][:5]
        ],
        "counter_hints": [
            {"paper": p["title"], "year": p["year"], "section_type": p["section_type"], "pages": _pages(p),
             "why": p["role"], "snippet": p["snippet"][:200]}
            for p in c["counter_hints"][:3]
        ],
        "grounding": c["grounding"],
        "signals": c["signals"],
    }


def _pages(x: dict) -> Optional[str]:
    a, b = x.get("page_start"), x.get("page_end")
    if not a:
        return None
    return f"{a}" if not b or b == a else f"{a}-{b}"
