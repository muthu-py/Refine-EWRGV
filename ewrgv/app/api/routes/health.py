"""
app/api/routes/health.py
-------------------------
Health check endpoint.

GET /api/v1/health

Returns application health status.  This is the only fully-functional
endpoint at the architecture stage.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.config import settings

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str
    timestamp: str


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check() -> HealthResponse:
    """
    Returns the application health status.

    Always returns HTTP 200 with status='ok' when the application is running.
    """
    return HealthResponse(
        status="ok",
        app=settings.APP_NAME,
        version=settings.APP_VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
