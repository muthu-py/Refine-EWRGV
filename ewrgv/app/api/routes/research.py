"""
app/api/routes/research.py
---------------------------
Research workflow placeholder endpoints.

These endpoints define the API contract for the EWRGV research workflow.
They currently return structured placeholder responses because the
underlying pipeline is not yet implemented.

Endpoints
---------
POST /api/v1/research/query     – Submit a research question
POST /api/v1/research/gaps      – Trigger gap detection for a query
POST /api/v1/research/validate  – Trigger EWRGV validation for a gap
GET  /api/v1/research/{job_id}  – Get status/results of a research job

Note: All business logic must live in the orchestration and domain layers,
NOT in these route handlers.
"""

from __future__ import annotations

from typing import Optional
from uuid import uuid4

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/research", tags=["Research"])


# ------------------------------------------------------------------ #
# Request / Response schemas (API-layer Pydantic models)
# NOTE: These are API schemas, NOT domain models.  Keep them separate.
# ------------------------------------------------------------------ #


class QueryRequest(BaseModel):
    query: str
    max_papers: int = 20
    top_k_gaps: int = 5


class QueryResponse(BaseModel):
    job_id: str
    status: str
    message: str


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
# Route handlers
# ------------------------------------------------------------------ #


@router.post("/query", response_model=QueryResponse, summary="Submit a research query")
async def submit_research_query(request: QueryRequest) -> QueryResponse:
    """
    Submit a research question to start the EWRGV pipeline.

    **NOT YET IMPLEMENTED** – returns a placeholder job_id.
    """
    job_id = str(uuid4())
    return QueryResponse(
        job_id=job_id,
        status="accepted",
        message=(
            "Research query accepted. Pipeline not yet implemented. "
            f"Query: '{request.query[:60]}...'"
        ),
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
