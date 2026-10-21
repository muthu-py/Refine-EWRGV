"""
app/api/routes/gaps_kg.py
---------------------------
Knowledge-graph and signal-based gap-detection endpoints (docs/gaps_kg_api.md).

    POST /research/kg/build              start a KG build job for a run
    GET  /research/kg/status/{job_id}    KG build job status
    GET  /research/kg/{run_id}           stored KG: summary / entities / statements /
                                         communities / papers / cytoscape / graphml
    POST /research/gaps                  start a gap-detection job for a run
    GET  /research/gaps/status/{job_id}  gap-detection job status
    GET  /research/gaps/{run_id}         stored candidates: summary / full / compare
"""

from __future__ import annotations

import asyncio
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/research", tags=["Knowledge graph & gaps"])


class KGBuildRequest(BaseModel):
    run_id: str
    query_id: Optional[str] = Field(None, description="Stored /query result; default: the run's linked query")
    relevance_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Fixed off-topic cut-off; default: run-relative (median similarity - 0.08, floor 0.55). 0 disables.",
    )


class GapDetectRequest(BaseModel):
    run_id: str
    query_id: Optional[str] = None
    detectors: list[Literal["explicit", "combination", "explicit_statements", "combination_sparsity",
                            "temporal", "temporal_staleness", "topics", "topic_density"]] = Field(
        default_factory=lambda: ["explicit", "combination", "temporal", "topics"])
    llm_wording: bool = Field(
        False, description="Optionally use the configured LLM to refine signal-based candidate wording."
    )
    include_llm_candidates: bool = Field(
        False, description="Optionally add grounded candidates proposed by the LLM to the candidate list."
    )
    include_llm_baseline: bool = Field(
        False, description="Optionally include a separate LLM baseline comparison in the GET response."
    )
    rebuild_kg: bool = Field(False, description="Rebuild the knowledge graph first")


class JobStarted(BaseModel):
    job_id: str
    run_id: str
    status: str
    message: str


class ValidateGapsRequest(BaseModel):
    run_id: str
    gap_ids: Optional[list[str]] = None
    sources: list[Literal["corpus", "openalex", "semantic_scholar"]] = Field(
        default_factory=lambda: ["corpus", "openalex", "semantic_scholar"])
    per_query: int = Field(10, ge=1, le=50)
    max_iterations: Optional[int] = Field(None, ge=1, le=5)


class AnalysisJobStatus(BaseModel):
    job_id: str
    run_id: str
    kind: str
    status: str
    progress: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    summary: dict = {}
    error: Optional[str] = None


async def _require_run(run_id: str) -> None:
    from app.storage.relational.database import get_db_pool  # noqa: PLC0415
    from app.storage.repositories.paper_repository import PaperRepository  # noqa: PLC0415

    if not run_id or not run_id.strip():
        raise HTTPException(status_code=400, detail="run_id must be a non-empty string.")
    if not await PaperRepository(await get_db_pool()).run_exists(run_id):
        raise HTTPException(status_code=404, detail=f"run_id '{run_id}' not found.")


async def _require_chunked_and_represented(run_id: str) -> None:
    """Reject analysis jobs unless this run has persisted chunk and embedding inputs."""
    from app.storage.relational.database import get_db_pool  # noqa: PLC0415

    pool = await get_db_pool()
    row = await pool.fetchrow(
        """SELECT
               COUNT(DISTINCT p.paper_id) FILTER (
                   WHERE p.chunk_status = 'CHUNKED'
               ) AS chunked,
               COUNT(DISTINCT p.paper_id) FILTER (
                   WHERE p.chunk_status = 'CHUNKED'
                     AND p.embedding_status IN ('EMBEDDED', 'PARTIAL')
               ) AS represented
           FROM papers p
           JOIN paper_sources ps ON ps.paper_id = p.paper_id
           WHERE ps.run_id = $1""",
        run_id,
    )
    chunked = int(row["chunked"] or 0) if row else 0
    represented = int(row["represented"] or 0) if row else 0
    if chunked < 1:
        raise HTTPException(
            status_code=409,
            detail=f"run_id '{run_id}' has no chunks. Run /chunks after /sections before /kg/build.",
        )
    if represented < 1:
        raise HTTPException(
            status_code=409,
            detail=f"run_id '{run_id}' has no represented chunks. Run /represent after /chunks before /kg/build.",
        )


async def _job_status(req: Request, job_id: str, kind: str) -> AnalysisJobStatus:
    job = await req.app.state.job_store.get_analysis_job(job_id)
    if job is None or job.kind != kind:
        raise HTTPException(status_code=404, detail=f"{kind} job '{job_id}' not found (jobs are kept in memory until restart).")
    return AnalysisJobStatus(**job.to_dict())


# ---------------------------------------------------------------------- #
# Knowledge graph
# ---------------------------------------------------------------------- #


@router.post("/kg/build", response_model=JobStarted, status_code=202, summary="Build the run's knowledge graph")
async def build_kg(payload: KGBuildRequest, req: Request) -> JobStarted:
    from app.knowledge_graph.jobs import run_kg_build  # noqa: PLC0415

    await _require_run(payload.run_id)
    await _require_chunked_and_represented(payload.run_id)
    job = await req.app.state.job_store.create_analysis_job(payload.run_id, "kg_build")
    asyncio.create_task(run_kg_build(job, req.app.state.job_store, payload.query_id, payload.relevance_threshold))
    return JobStarted(job_id=job.job_id, run_id=payload.run_id, status=job.status.value,
                      message=f"Knowledge-graph build started. Poll GET /api/v1/research/kg/status/{job.job_id}.")


@router.get("/kg/status/{job_id}", response_model=AnalysisJobStatus, summary="Knowledge-graph build status")
async def kg_status(job_id: str, req: Request) -> AnalysisJobStatus:
    return await _job_status(req, job_id, "kg_build")


@router.get("/kg/{run_id}", summary="View the run's knowledge graph")
async def get_kg(
    run_id: str,
    view: Literal["summary", "entities", "statements", "claims", "communities", "papers", "cytoscape", "graphml"] = "summary",
    entity_type: Optional[str] = Query(None, description="Filter for view=entities, e.g. METHOD, DATASET"),
    limit: int = Query(25, ge=1, le=1000),
):
    from app.knowledge_graph.graph import KnowledgeGraph  # noqa: PLC0415
    from app.knowledge_graph.jobs import kg_view  # noqa: PLC0415
    from app.storage.relational.database import get_db_pool  # noqa: PLC0415
    from app.storage.repositories.kg_repository import KnowledgeGraphRepository  # noqa: PLC0415

    stored = await KnowledgeGraphRepository(await get_db_pool()).get_graph(run_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"No knowledge graph for run_id '{run_id}'; POST /research/kg/build first.")
    if view == "graphml":
        xml = KnowledgeGraph.from_payload(stored["payload"]).to_graphml()
        return Response(content=xml, media_type="application/xml",
                        headers={"Content-Disposition": f'attachment; filename="kg_{run_id}.graphml"'})
    return kg_view(stored, view=view, entity_type=entity_type, limit=limit)


# ---------------------------------------------------------------------- #
# Gap detection
# ---------------------------------------------------------------------- #


@router.post("/gaps", response_model=JobStarted, status_code=202, summary="Detect candidate gaps (signal-based)")
async def detect_gaps(payload: GapDetectRequest, req: Request) -> JobStarted:
    from app.core.config import settings
    from app.core.exceptions import ServiceUnavailableError
    from app.knowledge_graph.jobs import run_gap_detection  # noqa: PLC0415

    if not settings.RERANK_ENABLED:
        raise ServiceUnavailableError(
            "Cross-encoder reranking is required for /research/gaps; set RERANK_ENABLED=true."
        )
    if payload.llm_wording or payload.include_llm_candidates or payload.include_llm_baseline:
        provider = (settings.LLM_PROVIDER or "openrouter").lower().strip()
        has_key = bool(settings.LLM_API_KEY) or (
            provider in ("openrouter", "open_router") and bool(settings.OPENROUTER_API_KEY)
        )
        if not has_key:
            raise ServiceUnavailableError(
                "LLM-assisted gap options require a configured API key for the selected LLM provider."
            )
    await _require_run(payload.run_id)
    await _require_chunked_and_represented(payload.run_id)
    from app.storage.relational.database import get_db_pool  # noqa: PLC0415
    from app.storage.repositories.kg_repository import KnowledgeGraphRepository  # noqa: PLC0415
    if await KnowledgeGraphRepository(await get_db_pool()).get_graph(payload.run_id) is None:
        raise HTTPException(
            status_code=409,
            detail=f"No knowledge graph for run_id '{payload.run_id}'. Run /kg/build after /chunks and /represent before /gaps.",
        )
    job = await req.app.state.job_store.create_analysis_job(payload.run_id, "gap_detection")
    asyncio.create_task(run_gap_detection(
        job, req.app.state.job_store, query_id=payload.query_id, detectors=list(payload.detectors),
        llm_wording=payload.llm_wording, include_llm_baseline=payload.include_llm_baseline,
        include_llm_candidates=payload.include_llm_candidates, rebuild_kg=payload.rebuild_kg,
    ))
    return JobStarted(job_id=job.job_id, run_id=payload.run_id, status=job.status.value,
                      message=f"Gap detection started. Poll GET /api/v1/research/gaps/status/{job.job_id}.")


@router.get("/gaps/status/{job_id}", response_model=AnalysisJobStatus, summary="Gap-detection job status")
async def gaps_status(job_id: str, req: Request) -> AnalysisJobStatus:
    return await _job_status(req, job_id, "gap_detection")


@router.get("/gaps/{run_id}", summary="View detected candidate gaps")
async def get_gaps(
    run_id: str,
    view: Literal["summary", "full", "compare"] = "summary",
    top: int = Query(10, ge=1, le=200),
    detector: Optional[Literal["explicit_statements", "combination_sparsity", "temporal_staleness", "topic_density"]] = None,
):
    from app.knowledge_graph.jobs import gaps_view  # noqa: PLC0415
    from app.storage.relational.database import get_db_pool  # noqa: PLC0415
    from app.storage.repositories.kg_repository import KnowledgeGraphRepository  # noqa: PLC0415

    stored = await KnowledgeGraphRepository(await get_db_pool()).get_gaps(run_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"No gap results for run_id '{run_id}'; POST /research/gaps first.")
    return gaps_view(stored, view=view, top=top, detector=detector)


@router.post("/validate", response_model=JobStarted, status_code=202, summary="Validate candidate gaps")
async def validate_gaps(payload: ValidateGapsRequest, req: Request) -> JobStarted:
    from app.validation.jobs import run_gap_validation  # noqa: PLC0415
    from app.storage.relational.database import get_db_pool  # noqa: PLC0415
    from app.storage.repositories.kg_repository import KnowledgeGraphRepository  # noqa: PLC0415

    await _require_run(payload.run_id)
    if await KnowledgeGraphRepository(await get_db_pool()).get_gaps(payload.run_id) is None:
        raise HTTPException(
            status_code=409,
            detail=f"No candidate gaps for run_id '{payload.run_id}'. Run /gaps after /kg/build before /validate.",
        )
    if not payload.sources:
        raise HTTPException(status_code=422, detail="Select at least one validation source.")
    job = await req.app.state.job_store.create_analysis_job(payload.run_id, "gap_validation")
    asyncio.create_task(run_gap_validation(
        job, req.app.state.job_store, gap_ids=payload.gap_ids, sources=tuple(payload.sources),
        per_query=payload.per_query, max_iterations=payload.max_iterations,
    ))
    return JobStarted(job_id=job.job_id, run_id=payload.run_id, status=job.status.value,
                      message=f"Gap validation started. Poll GET /api/v1/research/validate/status/{job.job_id}.")


@router.get("/validate/status/{job_id}", response_model=AnalysisJobStatus, summary="Gap-validation job status")
async def validation_status(job_id: str, req: Request) -> AnalysisJobStatus:
    return await _job_status(req, job_id, "gap_validation")


@router.get("/validate/{run_id}", summary="View stored gap validation results")
async def get_validation(run_id: str):
    from app.storage.relational.database import get_db_pool  # noqa: PLC0415
    from app.storage.repositories.kg_repository import KnowledgeGraphRepository  # noqa: PLC0415

    stored = await KnowledgeGraphRepository(await get_db_pool()).get_validations(run_id)
    if stored is None:
        raise HTTPException(status_code=404, detail=f"No validation results for run_id '{run_id}'; POST /research/validate first.")
    return stored
