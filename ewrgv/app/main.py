"""
app/main.py
-----------
FastAPI application entry point for the EWRGV system.

Responsibilities
----------------
* Create the FastAPI application instance.
* Configure startup/shutdown lifecycle (logging, configuration validation).
* Register API routers.
* Register global exception handlers.
* Expose the WSGI/ASGI callable for uvicorn.

The application layer is deliberately thin.  Business logic lives in
orchestration/, domain/, and their dependencies.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import health, research
from app.core.config import settings
from app.core.exceptions import APIError, EWRGVError
from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------------ #
# Lifespan (startup / shutdown)
# ------------------------------------------------------------------ #


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan handler.

    Startup:  Configure logging, validate settings, log ready message.
    Shutdown: Graceful teardown (connection pools, etc.) – to be added.
    """
    configure_logging(settings.LOG_LEVEL)
    logger.info("Starting %s v%s", settings.APP_NAME, settings.APP_VERSION)
    logger.info("Debug mode: %s", settings.DEBUG)

    yield  # Application runs here

    logger.info("Shutting down %s", settings.APP_NAME)


# ------------------------------------------------------------------ #
# Application factory
# ------------------------------------------------------------------ #


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns a configured FastAPI instance ready to be served by uvicorn.
    """
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "Evidence-Weighted Research Gap Validation (EWRGV) API. "
            "An end-to-end RAG-based research gap identification and validation system."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ---- CORS ---- #
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ---- Exception handlers ---- #
    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError) -> JSONResponse:
        logger.warning("APIError: %s (status=%d)", exc.message, exc.status_code)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": exc.message, "detail": exc.detail},
        )

    @app.exception_handler(EWRGVError)
    async def ewrgv_error_handler(request: Request, exc: EWRGVError) -> JSONResponse:
        logger.error("Unhandled EWRGVError: %s", exc.message, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "detail": exc.message},
        )

    # ---- Routers ---- #
    app.include_router(health.router, prefix=settings.API_V1_PREFIX)
    app.include_router(research.router, prefix=settings.API_V1_PREFIX)

    return app


# ------------------------------------------------------------------ #
# Application instance (for uvicorn: uvicorn app.main:app)
# ------------------------------------------------------------------ #

app = create_app()
