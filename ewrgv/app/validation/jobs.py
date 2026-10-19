"""Background orchestration and persistence for Phase 4 gap validation."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Optional

from app.core.job_store import AnalysisJob, JobStatus
from app.core.logging import get_logger

logger = get_logger(__name__)


async def run_gap_validation(job: AnalysisJob, store: Any, gap_ids: Optional[list[str]] = None,
                             sources: tuple[str, ...] = ("corpus", "openalex", "semantic_scholar"),
                             per_query: int = 10, max_iterations: Optional[int] = None) -> None:
    from app.providers.embeddings.factory import create_embedding_provider
    from app.retrieval.service import RetrievalService
    from app.storage.relational.database import get_db_pool
    from app.storage.repositories.kg_repository import KnowledgeGraphRepository
    from app.validation.sources import ExternalLiterature
    from app.validation.validator import EWRGVValidator
    async def set_state(**changes: Any) -> None:
        for key, value in changes.items():
            setattr(job, key, value)
        await store.update_analysis_job(job)

    external = ExternalLiterature() if set(sources) & {"openalex", "semantic_scholar"} else None
    try:
        await set_state(status=JobStatus.RUNNING, started_at=datetime.now(timezone.utc), progress="loading candidate gaps")
        repo = KnowledgeGraphRepository(await get_db_pool())
        graph_row, gaps_row = await asyncio.gather(repo.get_graph(job.run_id), repo.get_gaps(job.run_id))
        if graph_row is None or gaps_row is None:
            raise LookupError("Build the knowledge graph and run gap detection before validation.")
        from app.knowledge_graph.graph import KnowledgeGraph
        kg = KnowledgeGraph.from_payload(graph_row["payload"])
        available = gaps_row["candidates"]
        if gap_ids is not None:
            requested = set(gap_ids)
            candidates = [candidate for candidate in available if candidate.get("gap_id") in requested]
            found = {candidate.get("gap_id") for candidate in candidates}
            missing = requested - found
            if missing:
                raise LookupError(f"Unknown gap_id(s) for this run: {', '.join(sorted(missing))}")
        else:
            candidates = available[:10]
        if not candidates:
            raise ValueError("There are no candidate gaps to validate.")

        embedder = create_embedding_provider()
        from app.knowledge_graph.jobs import load_corpus
        corpus = await load_corpus(job.run_id, embedder)
        retrieval = RetrievalService(corpus, embedder) if "corpus" in sources else None
        validator = EWRGVValidator(kg, corpus, embedder, retrieval=retrieval, external=external,
                                   sources=sources, per_query=per_query, max_iterations=max_iterations)
        views = []
        for index, candidate in enumerate(candidates, start=1):
            await set_state(progress=f"validating candidate {index}/{len(candidates)}")
            _, view = await validator.validate_candidate(candidate)
            views.append(view)
        unavailable = sorted(validator.unavailable_sources)
        summary = {"validated": len(views), "classifications": {}, "unavailable_sources": unavailable,
                   "external_api_calls": external.calls if external else {}}
        for view in views:
            label = view["classification"]
            summary["classifications"][label] = summary["classifications"].get(label, 0) + 1
        await repo.save_validations(job.run_id, {"sources": list(sources), "per_query": per_query,
                                                  "max_iterations": max_iterations}, summary, views)
        await set_state(status=JobStatus.COMPLETED, progress="done", summary=summary)
    except Exception as exc:  # noqa: BLE001 - job errors are returned by status endpoint
        logger.error("Gap validation failed for run %s: %s", job.run_id, exc, exc_info=True)
        await set_state(status=JobStatus.FAILED, progress="failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if external is not None:
            await external.close()
        job.completed_at = datetime.now(timezone.utc)
        await store.update_analysis_job(job)
