"""
app/api/routes/research.py
---------------------------
Research workflow endpoints for the EWRGV API.

Endpoints
---------
POST /api/v1/research/query               – Submit a research question; returns
                                             structured understanding + expanded queries
POST /api/v1/research/collect             – Search academic literature
POST /api/v1/research/acquire             – Start full-text acquisition
POST /api/v1/research/retrieve/dense      – Dense (semantic) retrieval
POST /api/v1/research/retrieve/sparse     – Sparse (BM25) retrieval
POST /api/v1/research/retrieve/hybrid     – Hybrid (RRF fusion) retrieval
POST /api/v1/research/gaps               – Trigger gap detection for a query
GET  /api/v1/research/{job_id}           – Get status/results of a research job

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

from functools import lru_cache
from typing import Any, Literal, Optional
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
    saved: bool = False   # stored in query_runs, so /collect {query_id} works


class CollectRequest(BaseModel):
    """
    Request body for POST /research/collect.

    Normal use: pass the ``query_id`` returned by /query; the stored question
    and ALL its expanded queries are searched.  ``query`` / ``expanded_queries``
    still work without a query_id (and override the stored ones when given).
    """
    query_id: Optional[str] = None
    query: Optional[str] = None
    expanded_queries: list[str] = []
    include_original_query: bool = True     # also search the question itself
    max_results_per_query: Optional[int] = None   # default: SEMANTIC_SCHOLAR_MAX_RESULTS
    max_total_results: Optional[int] = None       # default: LITERATURE_MAX_TOTAL_RESULTS
    max_queries: Optional[int] = None              # default: LITERATURE_MAX_QUERIES


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
    query_id: Optional[str] = None
    capped: bool = False
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


class RetrieveFilterFields(BaseModel):
    """Optional result filters shared by every retrieve endpoint."""
    section_types: list[str] = []      # e.g. ["LIMITATIONS"] or the preset ["gap_evidence"]
    paper_ids: list[str] = []
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    include_noise: bool = False        # True also searches references / acknowledgments / boilerplate


class RetrieveRequest(RetrieveFilterFields):
    """Request body for POST /research/retrieve/{dense,sparse,hybrid}."""
    run_id: str
    query: str
    top_k: int = 10


class MultiRetrieveRequest(RetrieveFilterFields):
    """
    Request body for POST /research/retrieve/multi.

    Give ``queries`` explicitly, or a ``query_id`` from /query to use the
    stored question plus all its expanded queries.
    """
    run_id: str
    queries: list[str] = []
    query_id: Optional[str] = None
    method: str = "hybrid"
    top_k: int = 10


class RetrieveResultItem(BaseModel):
    """Single result in the retrieval response."""
    chunk_id: str
    paper_id: str
    section: str = ""
    section_type: str = ""
    page_start: int = 0
    page_end: int = 0
    text: str = ""
    score: float = 0.0
    rank: int = 0
    retrieval_method: str = ""
    metadata: dict = {}


class RetrieveResponse(BaseModel):
    """Response from POST /research/retrieve/* endpoints."""
    run_id: str
    query: str
    top_k: int
    total_results: int
    results: list[RetrieveResultItem]


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


@lru_cache(maxsize=1)
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
                max_retries=settings.LITERATURE_MAX_RETRIES,
                retry_base_delay=settings.LITERATURE_RETRY_BASE_DELAY,
                min_request_interval=settings.LITERATURE_MIN_REQUEST_INTERVAL,
                max_retry_delay=settings.LITERATURE_MAX_RETRY_DELAY,
                cache_ttl=settings.LITERATURE_SEARCH_CACHE_TTL,
            )
        elif name == "openalex":
            providers[name] = OpenAlexCollector(
                base_url=settings.OPENALEX_BASE_URL,
                mailto=settings.OPENALEX_EMAIL,
                timeout=settings.LITERATURE_REQUEST_TIMEOUT,
                max_retries=settings.LITERATURE_MAX_RETRIES,
                retry_base_delay=settings.LITERATURE_RETRY_BASE_DELAY,
                min_request_interval=settings.LITERATURE_MIN_REQUEST_INTERVAL,
                max_retry_delay=settings.LITERATURE_MAX_RETRY_DELAY,
                cache_ttl=settings.LITERATURE_SEARCH_CACHE_TTL,
            )

    if not providers:
        return None

    return LiteratureCollectionService(
        providers=providers,
        max_results_per_query=settings.SEMANTIC_SCHOLAR_MAX_RESULTS,
        max_total_results=settings.LITERATURE_MAX_TOTAL_RESULTS,
        max_queries=settings.LITERATURE_MAX_QUERIES,
    )


def _search_queries(question: str, expanded: list[str], include_original: bool = True) -> list[str]:
    """The question first (unless excluded), then the expansions; case-insensitive de-duplication."""
    seen: set[str] = set()
    out: list[str] = []
    for q in ([question] if include_original else []) + list(expanded or []):
        key = " ".join((q or "").lower().split())
        if key and key not in seen:
            seen.add(key)
            out.append(q.strip())
    return out


async def _save_query_run(query: Any) -> bool:
    """Persist a /query result; False (logged) when the database is unavailable."""
    try:
        from app.storage.relational.database import get_db_pool  # noqa: PLC0415
        from app.storage.repositories.query_repository import QueryRunRepository  # noqa: PLC0415

        await QueryRunRepository(await get_db_pool()).save(query)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to persist query run", extra={"query_id": query.query_id, "error": str(exc)})
        return False


async def _load_query_run(query_id: str) -> Optional[dict]:
    from app.storage.relational.database import get_db_pool  # noqa: PLC0415
    from app.storage.repositories.query_repository import QueryRunRepository  # noqa: PLC0415

    return await QueryRunRepository(await get_db_pool()).get(query_id)


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
        # Stub mode — no LLM configured; still stored so /collect {query_id} works
        stub = ResearchQuery(query=request.query)
        job_id = stub.query_id
        saved = await _save_query_run(stub)
        return QueryResponse(
            job_id=job_id,
            query_id=job_id,
            saved=saved,
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

    saved = await _save_query_run(enriched)
    return QueryResponse(
        job_id=enriched.query_id,
        query_id=enriched.query_id,
        saved=saved,
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
    Search academic literature for a research question.

    Pass the ``query_id`` from /query: the stored question **and all of its
    expanded queries** are searched on every configured provider (Semantic
    Scholar, OpenAlex).  Papers are persisted with provenance and the run is
    linked to the query.  Returns ``run_id`` for /acquire and later stages.

    Errors: 400 without query/query_id, 404 for an unknown query_id,
    503 when the run cannot be saved (no run_id is returned in that case).
    """
    import asyncio  # noqa: PLC0415

    from app.core.exceptions import BadRequestError, NotFoundError, ServiceUnavailableError  # noqa: PLC0415
    from app.domain.models.query import ResearchQuery  # noqa: PLC0415

    question = (request.query or "").strip()
    expanded = [q for q in request.expanded_queries if q and q.strip()]
    if request.query_id:
        try:
            stored = await _load_query_run(request.query_id)
        except Exception as exc:  # noqa: BLE001
            raise ServiceUnavailableError(f"Could not load query_id '{request.query_id}': {exc}") from exc
        if stored is None:
            raise NotFoundError(f"query_id '{request.query_id}' not found; call /query first.")
        question = question or stored["research_question"]
        expanded = expanded or list(stored["expanded_queries"])
    if not question:
        raise BadRequestError("Give a query_id from /query, or a query.")
    for name, value in (("max_results_per_query", request.max_results_per_query),
                        ("max_total_results", request.max_total_results),
                        ("max_queries", request.max_queries)):
        if value is not None and value < 1:
            raise BadRequestError(f"{name} must be >= 1.")

    service = _get_collection_service()
    if service is None:
        return CollectResponse(
            status="error",
            query_id=request.query_id,
            provider_errors=[{"provider": "all", "error": "No literature providers configured."}],
        )

    rq = ResearchQuery(
        query=question,
        expanded_queries=_search_queries(question, expanded, request.include_original_query),
    )
    # Provider calls are blocking HTTP; keep the event loop free.
    result = await asyncio.to_thread(
        service.collect,
        rq,
        max_results_per_query=request.max_results_per_query,
        max_total_results=request.max_total_results,
        max_queries=request.max_queries,
    )

    # Persist the run and its papers; without this the run_id is useless.
    try:
        from app.storage.relational.database import get_db_pool  # noqa: PLC0415
        from app.storage.repositories.paper_repository import PaperRepository  # noqa: PLC0415

        pool = await get_db_pool()
        await PaperRepository(pool).save_collection_result(result, question, query_id=request.query_id)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to persist collection result to Supabase",
            extra={"error": str(exc), "query_id": rq.query_id},
        )
        raise ServiceUnavailableError(f"Collection succeeded but could not be saved: {exc}") from exc

    try:
        from app.ingestion.deduplication.service import Deduplicator  # noqa: PLC0415

        stats = await Deduplicator(pool).run()
        logger.info("Deduplication after collection", extra=stats)
    except Exception as exc:  # noqa: BLE001 - the run is saved; duplicates can be merged later
        logger.error("Deduplication after collection failed", extra={"error": str(exc)})

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
        query_id=request.query_id,
        capped=result.capped,
        queries_searched=result.queries_searched,
        providers_used=result.providers_used,
        total_results=result.total_collected,
        papers=papers_resp,
        provider_errors=result.provider_errors,
    )


@router.get("/papers/{run_id}", summary="List papers collected for a run")
async def list_run_papers(
    run_id: str,
    q: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    sort_by: Literal["downloaded", "year_desc", "year_asc", "title_asc", "title_desc"] = "downloaded",
) -> dict:
    """Return searchable, paginated paper metadata and pipeline states."""
    from fastapi import HTTPException  # noqa: PLC0415
    from app.storage.relational.database import get_db_pool  # noqa: PLC0415
    from app.storage.repositories.paper_repository import PaperRepository  # noqa: PLC0415

    if limit < 1 or limit > 100 or offset < 0:
        raise HTTPException(status_code=400, detail="limit must be 1..100 and offset must be >= 0.")
    papers, total = await PaperRepository(await get_db_pool()).list_run_papers(
        run_id, limit=limit, offset=offset, search=q, sort_by=sort_by,
    )
    return {"run_id": run_id, "total": total, "limit": limit, "offset": offset, "papers": papers}


@router.get("/runs", summary="List recent research runs")
async def list_research_runs(limit: int = 30) -> dict:
    """List recent run records for the workspace dashboard."""
    from fastapi import HTTPException  # noqa: PLC0415
    from app.storage.relational.database import get_db_pool  # noqa: PLC0415
    from app.storage.repositories.paper_repository import PaperRepository  # noqa: PLC0415

    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be 1..100.")
    runs = await PaperRepository(await get_db_pool()).list_runs(limit=limit)
    return {"runs": runs}


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


def _build_retrieve_response(request: RetrieveRequest, results: list) -> RetrieveResponse:
    """Map domain RetrievalResult list to the API response schema."""
    return RetrieveResponse(
        run_id=request.run_id,
        query=request.query,
        top_k=request.top_k,
        total_results=len(results),
        results=[
            RetrieveResultItem(
                chunk_id=r.chunk_id,
                paper_id=r.paper_id,
                section=r.section,
                section_type=r.section_type,
                page_start=r.page_start,
                page_end=r.page_end,
                text=r.text,
                score=r.score,
                rank=r.rank,
                retrieval_method=r.retrieval_method,
                metadata=r.metadata,
            )
            for r in results
        ],
    )


def _validate_retrieve_request(request: RetrieveRequest) -> None:
    """Shared validation for all retrieve endpoints."""
    from app.core.exceptions import BadRequestError  # noqa: PLC0415

    if not request.run_id or not request.run_id.strip():
        raise BadRequestError("run_id must be a non-empty string.")
    if not request.query or not request.query.strip():
        raise BadRequestError("Query must be a non-empty string.")
    if not isinstance(request.top_k, int) or request.top_k < 1 or request.top_k > 100:
        raise BadRequestError("top_k must be an integer between 1 and 100.")


async def _build_retrieval_service(run_id: str, with_dense: bool) -> Any:
    """
    Load the run's corpus (PaperChunks, embeddings, BM25 index) and build
    a RetrievalService over it.

    Dense retrieval is enabled when an embedding provider is configured.
    The corpus is cached per run (see app.retrieval.corpus_cache).
    """
    from fastapi import HTTPException  # noqa: PLC0415

    from app.core.config import settings  # noqa: PLC0415
    from app.ingestion.corpus.run_corpus import RunCorpusLoader  # noqa: PLC0415
    from app.retrieval.service import RetrievalService  # noqa: PLC0415
    from app.storage.relational.database import get_db_pool  # noqa: PLC0415
    from app.storage.repositories.embedding_repository import EmbeddingRepository  # noqa: PLC0415
    from app.storage.repositories.paper_repository import PaperRepository  # noqa: PLC0415
    from app.storage.supabase_storage import SupabaseStorageClient  # noqa: PLC0415

    pool = await get_db_pool()
    paper_repo = PaperRepository(pool)
    if not await paper_repo.run_exists(run_id):
        raise HTTPException(status_code=404, detail=f"run_id '{run_id}' not found.")

    embedding_provider = None
    from app.providers.embeddings.factory import (  # noqa: PLC0415
        create_embedding_provider,
        embedding_provider_configured,
    )

    if with_dense and embedding_provider_configured():

        embedding_provider = create_embedding_provider()

    from app.retrieval import corpus_cache  # noqa: PLC0415

    loader = RunCorpusLoader(
        paper_repo=paper_repo,
        storage_client=SupabaseStorageClient(),
        embedding_repo=EmbeddingRepository(pool) if embedding_provider is not None else None,
    )
    model = embedding_provider.get_model_name() if embedding_provider is not None else None
    corpus = await corpus_cache.get_or_load(run_id, model, lambda: loader.load(run_id))
    if not corpus.get_all_chunks():
        raise HTTPException(
            status_code=409,
            detail=(
                f"run_id '{run_id}' has no chunked papers yet; run /chunks "
                "(and /represent for dense retrieval) first."
            ),
        )
    return RetrievalService(corpus, embedding_provider=embedding_provider)


def _filters_from(request: RetrieveFilterFields) -> Any:
    from app.core.exceptions import BadRequestError  # noqa: PLC0415
    from app.retrieval.filters import RetrievalFilters  # noqa: PLC0415

    if request.year_from is not None and request.year_to is not None and request.year_from > request.year_to:
        raise BadRequestError("year_from must be <= year_to.")
    return RetrievalFilters.build(
        section_types=request.section_types,
        paper_ids=request.paper_ids,
        year_from=request.year_from,
        year_to=request.year_to,
        exclude_noise=not request.include_noise,
    )


async def _run_retrieval(request: RetrieveRequest, method: str) -> RetrieveResponse:
    _validate_retrieve_request(request)
    filters = _filters_from(request)
    service = await _build_retrieval_service(request.run_id, with_dense=method != "sparse")
    results = await service.aretrieve(
        query=request.query, top_k=request.top_k, method=method, filters=filters
    )
    return _build_retrieve_response(request, results)


@router.post(
    "/retrieve/dense",
    response_model=RetrieveResponse,
    summary="Dense (semantic) retrieval",
)
async def retrieve_dense(request: RetrieveRequest) -> RetrieveResponse:
    """
    Run semantic dense retrieval over the run's chunks.

    Embeds the query and ranks the run's chunks by cosine similarity to
    their stored embeddings (from /represent).  Requires a configured embedding provider.
    """
    return await _run_retrieval(request, "dense")


@router.post(
    "/retrieve/sparse",
    response_model=RetrieveResponse,
    summary="Sparse (BM25) retrieval",
)
async def retrieve_sparse(request: RetrieveRequest) -> RetrieveResponse:
    """
    Run BM25 keyword retrieval over the run's chunks.

    Uses the run's persisted BM25 index (from /represent), or builds one
    from the chunks if it is missing or stale (k1=1.5, b=0.75).
    """
    return await _run_retrieval(request, "sparse")


@router.post(
    "/retrieve/hybrid",
    response_model=RetrieveResponse,
    summary="Hybrid (fusion) retrieval",
)
async def retrieve_hybrid(request: RetrieveRequest) -> RetrieveResponse:
    """
    Fuse dense and BM25 retrieval over the run's chunks with Reciprocal
    Rank Fusion (k=60).  Falls back to BM25 alone when no embedding API
    key is configured.
    """
    return await _run_retrieval(request, "hybrid")


@router.post(
    "/retrieve/multi",
    response_model=RetrieveResponse,
    summary="Multi-query retrieval (question + expansions, RRF-fused)",
)
async def retrieve_multi(request: MultiRetrieveRequest) -> RetrieveResponse:
    """
    Retrieve for several queries and fuse the ranked lists with Reciprocal
    Rank Fusion.  With ``query_id`` the stored question and all its expanded
    queries from /query are used.  Each result's metadata lists the queries
    that matched it (``matched_queries``).
    """
    from app.core.exceptions import BadRequestError, NotFoundError  # noqa: PLC0415

    if not request.run_id or not request.run_id.strip():
        raise BadRequestError("run_id must be a non-empty string.")
    if request.method not in ("dense", "sparse", "hybrid"):
        raise BadRequestError("method must be one of: dense, sparse, hybrid.")
    if not isinstance(request.top_k, int) or request.top_k < 1 or request.top_k > 100:
        raise BadRequestError("top_k must be an integer between 1 and 100.")

    queries = [q for q in request.queries if q and q.strip()]
    if request.query_id:
        stored = await _load_query_run(request.query_id)
        if stored is None:
            raise NotFoundError(f"query_id '{request.query_id}' not found.")
        queries = queries or _search_queries(stored["research_question"], stored["expanded_queries"])
    if not queries:
        raise BadRequestError("Give queries or a query_id.")

    filters = _filters_from(request)
    service = await _build_retrieval_service(request.run_id, with_dense=request.method != "sparse")
    results = await service.aretrieve_many(
        queries, top_k=request.top_k, method=request.method, filters=filters
    )
    return RetrieveResponse(
        run_id=request.run_id,
        query=" | ".join(queries),
        top_k=request.top_k,
        total_results=len(results),
        results=[
            RetrieveResultItem(
                chunk_id=r.chunk_id, paper_id=r.paper_id, section=r.section,
                section_type=r.section_type, page_start=r.page_start, page_end=r.page_end,
                text=r.text, score=r.score, rank=r.rank,
                retrieval_method=r.retrieval_method, metadata=r.metadata,
            )
            for r in results
        ],
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

