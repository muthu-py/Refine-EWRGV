"""
app/api/routes/research.py
---------------------------
Research workflow endpoints for the EWRGV API.

Endpoints
---------
POST /api/v1/research/query     – Submit a research question; returns
                                   structured understanding + expanded queries
POST /api/v1/research/gaps      – Trigger gap detection for a query
POST /api/v1/research/validate  – Trigger EWRGV validation for a gap
GET  /api/v1/research/{job_id}  – Get status/results of a research job

Design notes
------------
* API-layer Pydantic schemas (request/response) are kept separate from
  domain models.  Domain objects are not returned directly from routes.
* The /query endpoint now calls QueryUnderstandingService when an LLM API
  key is configured; it falls back to a stub response if the key is absent
  (useful for CI environments without API access).
* All business logic lives in the service layer, not here.

Note: All business logic must live in the orchestration and domain layers,
NOT in these route handlers.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/research", tags=["Research"])


# ------------------------------------------------------------------ #
# Request / Response schemas (API-layer Pydantic models)
# NOTE: These are API schemas, NOT domain models. Keep them separate.
# ------------------------------------------------------------------ #


class QueryRequest(BaseModel):
    query: str
    max_papers: int = 20
    top_k_gaps: int = 5


class QueryUnderstandingResponse(BaseModel):
    """Structured decomposition of the research question."""

    domain: Optional[str] = None
    problem: Optional[str] = None
    task: Optional[str] = None
    methods: list[str] = []
    concepts: list[str] = []
    entities: list[str] = []
    datasets: list[str] = []
    metrics: list[str] = []
    constraints: list[str] = []
    research_intent: Optional[str] = None
    terminology_variants: list[str] = []
    method_variants: list[str] = []


class QueryResponse(BaseModel):
    """
    Response from POST /api/v1/research/query.

    Fields
    ------
    job_id:
        Unique identifier for this research session (same as query_id).
    query_id:
        Unique identifier for the research query.
    status:
        "understood" when query understanding succeeded,
        "accepted" when the LLM is not configured (stub mode),
        "error" when query understanding failed.
    message:
        Human-readable status description.
    query_type:
        High-level classification of the research question.
    understanding:
        Structured decomposition of the research question.
        None when LLM is not configured.
    expanded_queries:
        Diverse search query strings for literature discovery.
        Empty when LLM is not configured.
    """

    job_id: str
    query_id: str
    status: str
    message: str
    query_type: str = "Unknown"
    understanding: Optional[QueryUnderstandingResponse] = None
    expanded_queries: list[str] = []


class CollectRequest(BaseModel):
    """Request body for POST /research/collect."""
    query: str
    expanded_queries: list[str] = []
    max_results_per_query: int = 20


class CollectPaperResponse(BaseModel):
    """Single paper in the collection response."""
    title: str
    authors: list[str] = []
    abstract: str = ""
    doi: Optional[str] = None
    source: str = ""
    source_query: str = ""
    paper_url: Optional[str] = None
    citation_count: Optional[int] = None
    publication_year: Optional[int] = None


class CollectResponse(BaseModel):
    """Response from POST /research/collect."""
    status: str
    run_id: str = ""
    queries_searched: list[str] = []
    providers_used: list[str] = []
    total_results: int = 0
    papers: list[CollectPaperResponse] = []
    provider_errors: list[dict] = []


class AcquireRequest(BaseModel):
    """Request body for POST /research/acquire."""
    run_id: str


class AcquireResponse(BaseModel):
    """Response from POST /research/acquire."""
    job_id: str
    run_id: str
    status: str
    message: str


# ---- Phase 2.5: Parsing schemas ---- #

class ParseRequest(BaseModel):
    """Request body for POST /research/parse."""
    run_id: str


class ParseResponse(BaseModel):
    """Response from POST /research/parse."""
    job_id: str
    run_id: str
    status: str
    message: str


class ParseStatusResponse(BaseModel):
    """Response from GET /research/parse/status/{job_id}."""
    job_id: str
    stage: str
    status: str
    result_summary: Optional[dict] = None


class GapsRequest(BaseModel):
    job_id: str


class GapsResponse(BaseModel):
    job_id: str
    status: str
    candidate_gaps: list[dict]


class ValidateRequest(BaseModel):
    job_id: str
    gap_ids: Optional[list[str]] = None   # None = validate all top-k


class ValidateResponse(BaseModel):
    job_id: str
    status: str
    validation_results: list[dict]


class JobStatusResponse(BaseModel):
    job_id: str
    stage: str
    status: str
    result_summary: Optional[dict] = None


class ChunkStatusResponse(BaseModel):
    job_id: str
    run_id: str
    status: str
    total_papers: int
    chunked: int
    failed: int
    error: Optional[str] = None


class RepresentationStartRequest(BaseModel):
    run_id: str


class RepresentationStartResponse(BaseModel):
    job_id: str
    run_id: str
    status: str
    message: str


class RepresentationStatusResponse(BaseModel):
    job_id: str
    run_id: str
    status: str
    total_papers: int
    processed_papers: int
    failed_papers: int
    total_chunks: int
    embedded_chunks: int
    failed_chunks: int
    embedding_status: str
    bm25_status: str
    error: Optional[str] = None


# ------------------------------------------------------------------ #
# Dependency: query understanding service (optional, lazy-instantiated)
# ------------------------------------------------------------------ #


def _get_qu_service() -> Any:
    """
    Lazily construct a QueryUnderstandingService if LLM is configured.

    Returns None if no API key is set — callers must handle this case
    and return a stub response.
    """
    from app.core.config import settings  # noqa: PLC0415

    has_key = bool(settings.LLM_API_KEY or settings.OPENROUTER_API_KEY)
    if not has_key:
        return None

    from app.providers.llm.factory import create_llm_provider  # noqa: PLC0415
    from app.query_understanding.service import QueryUnderstandingService  # noqa: PLC0415

    llm = create_llm_provider()
    return QueryUnderstandingService(
        llm=llm,
        max_expanded_queries=settings.QUERY_EXPANSION_MAX_QUERIES,
    )


def _get_collection_service() -> Any:
    """
    Lazily construct a LiteratureCollectionService from settings.

    Returns None if no literature providers are configured.
    """
    from app.core.config import settings  # noqa: PLC0415
    from app.providers.search.semantic_scholar import SemanticScholarCollector  # noqa: PLC0415
    from app.providers.search.openalex import OpenAlexCollector  # noqa: PLC0415
    from app.ingestion.collectors.service import LiteratureCollectionService  # noqa: PLC0415

    enabled = [
        p.strip()
        for p in settings.LITERATURE_PROVIDERS.split(",")
        if p.strip()
    ]
    if not enabled:
        return None

    providers: dict[str, Any] = {}
    for name in enabled:
        if name == "semantic_scholar":
            providers[name] = SemanticScholarCollector(
                api_key=settings.SEMANTIC_SCHOLAR_API_KEY,
                base_url=settings.SEMANTIC_SCHOLAR_BASE_URL,
                timeout=settings.LITERATURE_REQUEST_TIMEOUT,
            )
        elif name == "openalex":
            providers[name] = OpenAlexCollector(
                base_url=settings.OPENALEX_BASE_URL,
                mailto=settings.OPENALEX_EMAIL,
                timeout=settings.LITERATURE_REQUEST_TIMEOUT,
            )

    if not providers:
        return None

    return LiteratureCollectionService(
        providers=providers,
        max_results_per_query=settings.SEMANTIC_SCHOLAR_MAX_RESULTS,
        max_total_results=settings.LITERATURE_MAX_TOTAL_RESULTS,
    )


# ------------------------------------------------------------------ #
# Route handlers
# ------------------------------------------------------------------ #


@router.post("/query", response_model=QueryResponse, summary="Submit a research query")
async def submit_research_query(request: QueryRequest) -> QueryResponse:
    """
    Submit a research question to start the EWRGV pipeline.

    When an LLM API key is configured, this endpoint runs the full
    **Query Understanding & Expansion** step and returns:
    - A structured decomposition of the research question
    - Multiple expanded search queries for literature discovery

    When no LLM API key is configured (e.g., in CI), the endpoint
    returns a stub response with status "accepted".
    """
    from app.core.exceptions import ProviderError  # noqa: PLC0415
    from app.domain.models.query import ResearchQuery  # noqa: PLC0415

    qu_service = _get_qu_service()

    if qu_service is None:
        # Stub mode — no LLM configured
        job_id = str(uuid4())
        return QueryResponse(
            job_id=job_id,
            query_id=job_id,
            status="accepted",
            message=(
                "Research query accepted. LLM_API_KEY is not configured; "
                "query understanding is skipped. "
                f"Query: '{request.query[:80]}'"
            ),
        )

    # Run query understanding
    raw_query = ResearchQuery(query=request.query)
    try:
        enriched = qu_service.run(raw_query)
    except ProviderError as exc:
        return QueryResponse(
            job_id=raw_query.query_id,
            query_id=raw_query.query_id,
            status="error",
            message=f"Query understanding failed: {exc.message}",
        )

    # Map domain model → API response schema
    understanding_resp: Optional[QueryUnderstandingResponse] = None
    if enriched.understanding is not None:
        u = enriched.understanding
        understanding_resp = QueryUnderstandingResponse(
            domain=u.domain,
            problem=u.problem,
            task=u.task,
            methods=u.methods,
            concepts=u.concepts,
            entities=u.entities,
            datasets=u.datasets,
            metrics=u.metrics,
            constraints=u.constraints,
            research_intent=u.research_intent,
            terminology_variants=u.terminology_variants,
            method_variants=u.method_variants,
        )

    return QueryResponse(
        job_id=enriched.query_id,
        query_id=enriched.query_id,
        status="understood",
        message=(
            f"Query understanding complete. "
            f"{len(enriched.expanded_queries)} expanded queries generated."
        ),
        query_type=enriched.query_type.value if hasattr(enriched.query_type, 'value') else str(enriched.query_type),
        understanding=understanding_resp,
        expanded_queries=enriched.expanded_queries,
    )


@router.post("/collect", response_model=CollectResponse, summary="Collect literature")
async def collect_literature(request: CollectRequest) -> CollectResponse:
    """
    Search academic literature using the expanded queries from Phase 2.1.

    This endpoint dispatches to all configured providers (Semantic Scholar,
    OpenAlex) and returns normalised papers with provenance.
    """
    from app.domain.models.query import ResearchQuery  # noqa: PLC0415

    service = _get_collection_service()
    if service is None:
        return CollectResponse(
            status="error",
            provider_errors=[{"provider": "all", "error": "No literature providers configured."}],
        )

    rq = ResearchQuery(
        query=request.query,
        expanded_queries=request.expanded_queries,
    )
    result = service.collect(rq)

    # Persist the collected papers
    try:
        from app.storage.relational.database import get_db_pool
        from app.storage.repositories.paper_repository import PaperRepository
        
        pool = await get_db_pool()
        repo = PaperRepository(pool)
        await repo.save_collection_result(result, request.query)
        
        # Run deduplication on the corpus
        from app.ingestion.deduplication.service import Deduplicator  # noqa: PLC0415
        deduplicator = Deduplicator(pool)
        stats = await deduplicator.run()
        logger.info("Deduplication after collection", extra=stats)
    except Exception as exc:
        logger.error(
            "Failed to persist collection result to Supabase",
            extra={"error": str(exc), "query_id": rq.query_id}
        )

    papers_resp = []
    for cp in result.papers:
        papers_resp.append(CollectPaperResponse(
            title=cp.paper.title,
            authors=[a.name for a in cp.paper.authors],
            abstract=cp.paper.abstract[:500] if cp.paper.abstract else "",
            doi=cp.paper.doi,
            source=cp.provenance.source,
            source_query=cp.provenance.source_query,
            paper_url=cp.paper.paper_url,
            citation_count=cp.paper.citation_count,
            publication_year=cp.paper.publication_year,
        ))

    return CollectResponse(
        status=result.status,
        run_id=rq.query_id,
        queries_searched=result.queries_searched,
        providers_used=result.providers_used,
        total_results=result.total_collected,
        papers=papers_resp,
        provider_errors=result.provider_errors,
    )


@router.post("/acquire", response_model=AcquireResponse, summary="Start full-text acquisition")
async def start_acquisition(request: AcquireRequest, req: Request) -> AcquireResponse:
    """
    Start asynchronous full-text (PDF) acquisition for all papers in a
    research run that have not yet been processed.

    Returns immediately with a `job_id` that can be polled via
    `GET /api/v1/research/{job_id}`.
    """
    import asyncio  # noqa: PLC0415
    from app.storage.relational.database import get_db_pool  # noqa: PLC0415
    from app.storage.repositories.paper_repository import PaperRepository  # noqa: PLC0415
    from app.ingestion.acquisition.worker import run_acquisition  # noqa: PLC0415

    pool = await get_db_pool()
    repo = PaperRepository(pool)

    # Validate run_id exists
    if not await repo.run_exists(request.run_id):
        from fastapi import HTTPException  # noqa: PLC0415
        raise HTTPException(status_code=404, detail=f"run_id '{request.run_id}' not found.")

    job_store = req.app.state.job_store
    job = await job_store.create_job(request.run_id)

    # Fire-and-forget — returns immediately
    asyncio.create_task(
        run_acquisition(job=job, job_store=job_store, repo=repo)
    )

    logger.info(
        "Acquisition job created",
        extra={"job_id": job.job_id, "run_id": request.run_id},
    )

    return AcquireResponse(
        job_id=job.job_id,
        run_id=request.run_id,
        status=job.status.value,
        message=f"Acquisition job started. Poll GET /api/v1/research/{job.job_id} for progress.",
    )


@router.post("/gaps", response_model=GapsResponse, summary="Retrieve detected gaps")
async def get_detected_gaps(request: GapsRequest) -> GapsResponse:
    """
    Retrieve candidate gaps detected for a submitted query job.

    **NOT YET IMPLEMENTED** – returns an empty gap list.
    """
    return GapsResponse(
        job_id=request.job_id,
        status="not_implemented",
        candidate_gaps=[],
    )


@router.post(
    "/validate",
    response_model=ValidateResponse,
    summary="Trigger EWRGV validation",
)
async def trigger_validation(request: ValidateRequest) -> ValidateResponse:
    """
    Trigger EWRGV validation for the top-k candidate gaps of a job.

    **NOT YET IMPLEMENTED** – returns an empty validation result list.
    """
    return ValidateResponse(
        job_id=request.job_id,
        status="not_implemented",
        validation_results=[],
    )


@router.get(
    "/{job_id}",
    response_model=JobStatusResponse,
    summary="Get acquisition job status",
)
async def get_job_status(job_id: str, req: Request) -> JobStatusResponse:
    """
    Retrieve the current status and progress of an acquisition job.

    Returns real-time stats: acquired, failed, no_oa counts, and
    completion timestamp once the job finishes.
    """
    job_store = getattr(req.app.state, "job_store", None)
    if job_store is None:
        return JobStatusResponse(
            job_id=job_id,
            stage="not_found",
            status="No acquisition job found for this job_id.",
        )

    job = await job_store.get_job(job_id)

    if job is None:
        return JobStatusResponse(
            job_id=job_id,
            stage="not_found",
            status="No acquisition job found for this job_id.",
        )

    return JobStatusResponse(
        job_id=job.job_id,
        stage="acquisition",
        status=job.status.value,
        result_summary=job.to_dict(),
    )


# ================================================================== #
# Phase 2.5 — Document Parsing
# ================================================================== #


@router.post("/parse", response_model=ParseResponse, summary="Start document parsing")
async def start_parsing(request: ParseRequest, req: Request) -> ParseResponse:
    """
    Start asynchronous document parsing for all DOWNLOADED papers in a
    research run that have not yet been parsed.

    Returns immediately with a `job_id` that can be polled via
    `GET /api/v1/research/parse/status/{job_id}`.

    Only papers with full_text_status = 'DOWNLOADED' are eligible.
    Papers already in parsing_status = 'PARSED' are skipped (idempotent).
    """
    import asyncio  # noqa: PLC0415
    from app.storage.relational.database import get_db_pool  # noqa: PLC0415
    from app.storage.repositories.paper_repository import PaperRepository  # noqa: PLC0415
    from app.ingestion.parsers.worker import run_parsing  # noqa: PLC0415

    pool = await get_db_pool()
    repo = PaperRepository(pool)

    # Validate run_id exists
    if not await repo.run_exists(request.run_id):
        from fastapi import HTTPException  # noqa: PLC0415
        raise HTTPException(status_code=404, detail=f"run_id '{request.run_id}' not found.")

    job_store = req.app.state.job_store
    job = await job_store.create_parsing_job(request.run_id)

    # Fire-and-forget — returns immediately
    asyncio.create_task(
        run_parsing(job=job, job_store=job_store, repo=repo)
    )

    logger.info(
        "Parsing job created",
        extra={"job_id": job.job_id, "run_id": request.run_id},
    )

    return ParseResponse(
        job_id=job.job_id,
        run_id=request.run_id,
        status=job.status.value,
        message=(
            f"Parsing job started. "
            f"Poll GET /api/v1/research/parse/status/{job.job_id} for progress."
        ),
    )


@router.get(
    "/parse/status/{job_id}",
    response_model=ParseStatusResponse,
    summary="Get parsing job status",
)
async def get_parsing_status(job_id: str, req: Request) -> ParseStatusResponse:
    """
    Retrieve the current status and progress of a document parsing job.

    Returns real-time stats: parsed, failed, skipped counts, and
    completion timestamp once the job finishes.
    """
    job_store = getattr(req.app.state, "job_store", None)
    if job_store is None:
        return ParseStatusResponse(
            job_id=job_id,
            stage="not_found",
            status="No parsing job found for this job_id.",
        )

    job = await job_store.get_parsing_job(job_id)

    if job is None:
        return ParseStatusResponse(
            job_id=job_id,
            stage="not_found",
            status="No parsing job found for this job_id.",
        )

    return ParseStatusResponse(
        job_id=job.job_id,
        stage="parsing",
        status=job.status.value,
        result_summary=job.to_dict(),
    )


# ================================================================== #
# Phase 2.6 -- Section Extraction
# ================================================================== #


class SectionRequest(BaseModel):
    """Request body for POST /research/sections."""
    run_id: str


class SectionResponse(BaseModel):
    """Response from POST /research/sections."""
    job_id: str
    run_id: str
    status: str
    message: str


class SectionStatusResponse(BaseModel):
    """Response from GET /research/sections/status/{job_id}."""
    job_id: str
    stage: str
    status: str
    result_summary: Optional[dict] = None


@router.post("/sections", response_model=SectionResponse, summary="Start section extraction")
async def start_sectioning(request: SectionRequest, req: Request) -> SectionResponse:
    """
    Start asynchronous section extraction for all PARSED papers in a
    research run that have not yet been section-extracted.

    Returns immediately with a job_id that can be polled via
    GET /api/v1/research/sections/status/{job_id}.

    Only papers with parsing_status = PARSED are eligible.
    Papers already in section_status = EXTRACTED are skipped (idempotent).
    """
    import asyncio  # noqa: PLC0415
    from app.storage.relational.database import get_db_pool  # noqa: PLC0415
    from app.storage.repositories.paper_repository import PaperRepository  # noqa: PLC0415
    from app.ingestion.sectioning.worker import run_sectioning  # noqa: PLC0415

    pool = await get_db_pool()
    repo = PaperRepository(pool)

    if not await repo.run_exists(request.run_id):
        from fastapi import HTTPException  # noqa: PLC0415
        raise HTTPException(status_code=404, detail=f"run_id '{request.run_id}' not found.")

    job_store = req.app.state.job_store
    job = await job_store.create_sectioning_job(request.run_id)

    asyncio.create_task(
        run_sectioning(job=job, job_store=job_store, repo=repo)
    )

    logger.info(
        "Sectioning job created",
        extra={"job_id": job.job_id, "run_id": request.run_id},
    )

    return SectionResponse(
        job_id=job.job_id,
        run_id=request.run_id,
        status=job.status.value,
        message=(
            f"Sectioning job started. "
            f"Poll GET /api/v1/research/sections/status/{job.job_id} for progress."
        ),
    )


@router.get(
    "/sections/status/{job_id}",
    response_model=SectionStatusResponse,
    summary="Get section extraction job status",
)
async def get_sectioning_status(job_id: str, req: Request) -> SectionStatusResponse:
    """Retrieve the current status of a section extraction job."""
    job_store = getattr(req.app.state, "job_store", None)
    if job_store is None:
        return SectionStatusResponse(job_id=job_id, stage="not_found", status="No job store.")

    job = await job_store.get_sectioning_job(job_id)
    if job is None:
        return SectionStatusResponse(
            job_id=job_id, stage="not_found", status="No sectioning job found."
        )

    return SectionStatusResponse(
        job_id=job.job_id,
        stage="sectioning",
        status=job.status.value,
        result_summary=job.to_dict(),
    )


# ================================================================== #
# Phase 2.7 -- Chunking
# ================================================================== #


class ChunkRequest(BaseModel):
    """Request body for POST /research/chunks."""
    run_id: str


class ChunkResponse(BaseModel):
    """Response from POST /research/chunks."""
    job_id: str
    run_id: str
    status: str
    message: str


class ChunkStatusResponse(BaseModel):
    """Response from GET /research/chunks/status/{job_id}."""
    job_id: str
    stage: str
    status: str
    result_summary: Optional[dict] = None


@router.post("/chunks", response_model=ChunkResponse, summary="Start chunking")
async def start_chunking(request: ChunkRequest, req: Request) -> ChunkResponse:
    """
    Start asynchronous chunking for all EXTRACTED papers in a research run
    that have not yet been chunked.

    Returns immediately with a job_id that can be polled via
    GET /api/v1/research/chunks/status/{job_id}.

    Only papers with section_status = EXTRACTED are eligible.
    Papers already in chunk_status = CHUNKED are skipped (idempotent).
    """
    import asyncio  # noqa: PLC0415
    from app.storage.relational.database import get_db_pool  # noqa: PLC0415
    from app.storage.repositories.paper_repository import PaperRepository  # noqa: PLC0415
    from app.ingestion.chunking.worker import run_chunking  # noqa: PLC0415

    pool = await get_db_pool()
    repo = PaperRepository(pool)

    if not await repo.run_exists(request.run_id):
        from fastapi import HTTPException  # noqa: PLC0415
        raise HTTPException(status_code=404, detail=f"run_id '{request.run_id}' not found.")

    job_store = req.app.state.job_store
    job = await job_store.create_chunking_job(request.run_id)

    asyncio.create_task(
        run_chunking(job=job, job_store=job_store, repo=repo)
    )

    logger.info(
        "Chunking job created",
        extra={"job_id": job.job_id, "run_id": request.run_id},
    )

    return ChunkResponse(
        job_id=job.job_id,
        run_id=request.run_id,
        status=job.status.value,
        message=(
            f"Chunking job started. "
            f"Poll GET /api/v1/research/chunks/status/{job.job_id} for progress."
        ),
    )


@router.get(
    "/chunks/status/{job_id}",
    response_model=ChunkStatusResponse,
    summary="Get chunking job status",
)
async def get_chunking_status(job_id: str, req: Request) -> ChunkStatusResponse:
    """Retrieve the current status of a chunking job."""
    job_store = getattr(req.app.state, "job_store", None)
    if job_store is None:
        return ChunkStatusResponse(job_id=job_id, stage="not_found", status="No job store.")

    job = await job_store.get_chunking_job(job_id)
    if job is None:
        return ChunkStatusResponse(
            job_id=job_id, stage="not_found", status="No chunking job found."
        )

    return ChunkStatusResponse(
        job_id=job.job_id,
        stage="chunking",
        status=job.status.value,
        result_summary=job.to_dict(),
    )


# ================================================================== #
# Phase 2.8 — Corpus Health
# ================================================================== #


class CorpusHealthSummaryResponse(BaseModel):
    """Response from GET /research/corpus/health."""
    run_id: Optional[str] = None
    total_papers: int = 0
    downloaded: int = 0
    not_checked: int = 0
    no_oa_found: int = 0
    download_failed: int = 0
    invalid_document: int = 0
    parsed: int = 0
    sections_extracted: int = 0
    chunked: int = 0


class InconsistencyResponse(BaseModel):
    """A single detected DB / artifact inconsistency."""
    paper_id: str
    artifact_type: str
    db_status_field: str
    db_status_value: str
    artifact_path: str
    artifact_exists: bool
    kind: str
    description: str


class CorpusHealthReportResponse(BaseModel):
    """Response from GET /research/corpus/health/inconsistencies."""
    summary: CorpusHealthSummaryResponse
    inconsistencies: list[InconsistencyResponse] = []
    inconsistency_count: int = 0
    papers_checked: int = 0


@router.get(
    "/corpus/health",
    response_model=CorpusHealthSummaryResponse,
    summary="Get corpus health summary",
)
async def get_corpus_health(run_id: Optional[str] = None) -> CorpusHealthSummaryResponse:
    """
    Return aggregate corpus processing-status counts.

    Provides a quick overview of how many papers are at each stage of
    the processing pipeline (downloaded, parsed, sections extracted, chunked).

    Query Parameters
    ----------------
    run_id:
        Optional. If provided, restrict counts to papers in that research run.
        If omitted, returns corpus-wide counts.

    Note: This endpoint queries the database only (no storage I/O).
    """
    from app.storage.relational.database import get_db_pool  # noqa: PLC0415
    from app.storage.repositories.paper_repository import PaperRepository  # noqa: PLC0415

    try:
        pool = await get_db_pool()
        repo = PaperRepository(pool)
        counts = await repo.get_corpus_health_counts(run_id=run_id)
        return CorpusHealthSummaryResponse(
            run_id=run_id,
            total_papers=counts.get("total_papers", 0),
            downloaded=counts.get("downloaded", 0),
            not_checked=counts.get("not_checked", 0),
            no_oa_found=counts.get("no_oa_found", 0),
            download_failed=counts.get("download_failed", 0),
            invalid_document=counts.get("invalid_document", 0),
            parsed=counts.get("parsed", 0),
            sections_extracted=counts.get("sections_extracted", 0),
            chunked=counts.get("chunked", 0),
        )
    except Exception as exc:
        logger.error(
            "Corpus health summary failed",
            extra={"error": str(exc), "run_id": run_id},
        )
        return CorpusHealthSummaryResponse(run_id=run_id)


@router.get(
    "/corpus/health/inconsistencies",
    response_model=CorpusHealthReportResponse,
    summary="Detect corpus DB/artifact inconsistencies",
)
async def get_corpus_inconsistencies(
    run_id: Optional[str] = None,
) -> CorpusHealthReportResponse:
    """
    Detect mismatches between the DB processing status and actual storage artifacts.

    Checks situations such as:
    - DB says parsing_status = PARSED but parsed.json does not exist
    - DB says chunk_status   = CHUNKED but chunks.json does not exist
    - fulltext.pdf exists but full_text_status = NOT_CHECKED
    - parsed.json exists but parsing_status = NOT_PARSED

    This performs storage existence checks (HEAD requests) for each paper —
    for large corpora this may take a moment.

    Inconsistencies are REPORTED only, never auto-repaired.

    Query Parameters
    ----------------
    run_id:
        Optional. If provided, restrict checks to papers in that research run.
    """
    from app.storage.relational.database import get_db_pool  # noqa: PLC0415
    from app.storage.repositories.paper_repository import PaperRepository  # noqa: PLC0415
    from app.storage.document_storage import SupabaseDocumentStorage  # noqa: PLC0415
    from app.storage.corpus_health import CorpusHealthService  # noqa: PLC0415

    try:
        pool = await get_db_pool()
        repo = PaperRepository(pool)
        async with SupabaseDocumentStorage() as storage:
            service = CorpusHealthService(repo=repo, storage=storage)
            report = await service.get_full_report(run_id=run_id, check_artifacts=True)

        summary_resp = CorpusHealthSummaryResponse(
            run_id=run_id,
            total_papers=report.summary.total_papers,
            downloaded=report.summary.downloaded,
            not_checked=report.summary.not_checked,
            no_oa_found=report.summary.no_oa_found,
            download_failed=report.summary.download_failed,
            invalid_document=report.summary.invalid_document,
            parsed=report.summary.parsed,
            sections_extracted=report.summary.sections_extracted,
            chunked=report.summary.chunked,
        )
        inconsistency_resp = [
            InconsistencyResponse(
                paper_id=inc.paper_id,
                artifact_type=inc.artifact_type.value,
                db_status_field=inc.db_status_field,
                db_status_value=inc.db_status_value,
                artifact_path=inc.artifact_path,
                artifact_exists=inc.artifact_exists,
                kind=inc.kind.value,
                description=inc.description,
            )
            for inc in report.inconsistencies
        ]
        return CorpusHealthReportResponse(
            summary=summary_resp,
            inconsistencies=inconsistency_resp,
            inconsistency_count=report.inconsistency_count,
            papers_checked=report.papers_checked,
        )

    except Exception as exc:
        logger.error(
            "Corpus inconsistency check failed",
            extra={"error": str(exc), "run_id": run_id},
        )
        return CorpusHealthReportResponse(
            summary=CorpusHealthSummaryResponse(run_id=run_id),
        )


# ------------------------------------------------------------------ #
# Phase 2.9: Text Representation (Dense + Sparse)
# ------------------------------------------------------------------ #

@router.post("/represent", response_model=RepresentationStartResponse)
async def start_representation(
    request: Request,
    payload: RepresentationStartRequest,
) -> RepresentationStartResponse:
    """
    Start Phase 2.9 text representation for a research run.

    This creates both dense embeddings (pgvector) and a sparse BM25 index
    for all chunks associated with the given run.
    """
    import asyncio  # noqa: PLC0415
    from app.storage.relational.database import get_db_pool  # noqa: PLC0415
    from app.storage.repositories.paper_repository import PaperRepository  # noqa: PLC0415
    from app.storage.repositories.embedding_repository import EmbeddingRepository  # noqa: PLC0415
    from app.storage.supabase_storage import SupabaseStorageClient  # noqa: PLC0415
    from app.providers.embeddings.factory import create_embedding_provider  # noqa: PLC0415
    from app.ingestion.representation.service import RepresentationService  # noqa: PLC0415
    from app.ingestion.representation.worker import run_representation  # noqa: PLC0415

    job_store = request.app.state.job_store
    job = await job_store.create_representation_job(payload.run_id)

    try:
        pool = await get_db_pool()
        paper_repo = PaperRepository(pool)
        embedding_repo = EmbeddingRepository(pool)
        storage_client = SupabaseStorageClient()
        embedding_provider = create_embedding_provider()

        service = RepresentationService(
            embedding_provider=embedding_provider,
            embedding_repo=embedding_repo,
            paper_repo=paper_repo,
            storage_client=storage_client,
        )

        asyncio.create_task(run_representation(job, job_store, service))

        return RepresentationStartResponse(
            job_id=job.job_id,
            run_id=job.run_id,
            status=job.status.value,
            message="Representation process started in background",
        )
    except Exception as exc:
        job.status = JobStatus.FAILED
        job.error = str(exc)
        await job_store.update_representation_job(job)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start representation: {exc}",
        ) from exc


@router.get(
    "/represent/status/{job_id}",
    response_model=RepresentationStatusResponse,
)
async def get_representation_status(
    request: Request,
    job_id: str,
) -> RepresentationStatusResponse:
    """
    Check the status of a Phase 2.9 text representation background job.
    """
    from fastapi import HTTPException
    
    job_store = request.app.state.job_store
    job = await job_store.get_representation_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail=f"Representation job {job_id} not found")

    return RepresentationStatusResponse(
        job_id=job.job_id,
        run_id=job.run_id,
        status=job.status.value,
        total_papers=job.total_papers,
        processed_papers=job.processed_papers,
        failed_papers=job.failed_papers,
        total_chunks=job.total_chunks,
        embedded_chunks=job.embedded_chunks,
        failed_chunks=job.failed_chunks,
        embedding_status=job.embedding_status,
        bm25_status=job.bm25_status,
        error=job.error,
    )

