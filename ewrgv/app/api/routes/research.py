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

from fastapi import APIRouter, Depends
from pydantic import BaseModel

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
    queries_searched: list[str] = []
    providers_used: list[str] = []
    total_results: int = 0
    papers: list[CollectPaperResponse] = []
    provider_errors: list[dict] = []


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
        queries_searched=result.queries_searched,
        providers_used=result.providers_used,
        total_results=result.total_collected,
        papers=papers_resp,
        provider_errors=result.provider_errors,
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
    summary="Get research job status",
)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """
    Retrieve the current status and results of a research pipeline job.

    **NOT YET IMPLEMENTED** – returns placeholder status.
    """
    return JobStatusResponse(
        job_id=job_id,
        stage="not_implemented",
        status="Pipeline not yet implemented.",
    )
