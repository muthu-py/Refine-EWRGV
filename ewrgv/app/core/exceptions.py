"""
app/core/exceptions.py
-----------------------
Application-wide exception hierarchy for EWRGV.

Design principles
-----------------
* All custom exceptions inherit from ``EWRGVError`` so callers can catch the
  entire family with a single ``except EWRGVError`` clause.
* HTTP-aware variants carry a ``status_code`` so the FastAPI exception handler
  can map them to proper HTTP responses without inspecting the type.
* Business exceptions are kept separate from HTTP exceptions; the API layer
  translates them.
"""

from __future__ import annotations

from typing import Optional


# ------------------------------------------------------------------ #
# Base
# ------------------------------------------------------------------ #


class EWRGVError(Exception):
    """Root exception for all EWRGV application errors."""

    def __init__(self, message: str, detail: Optional[str] = None) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(message={self.message!r})"


# ------------------------------------------------------------------ #
# Domain / Business Exceptions
# ------------------------------------------------------------------ #


class ConfigurationError(EWRGVError):
    """Raised when required configuration is missing or invalid."""


class ProviderError(EWRGVError):
    """Raised when an external provider (LLM, embedding, search) fails."""


class EmbeddingAPIError(ProviderError):
    """
    The embedding API answered with an HTTP error.

    ``kind`` is one of "auth" (401/403), "rate_limit" (429), "bad_request"
    (400/404/422) or "upstream" (5xx / anything else).  ``provider_message``
    is the short message parsed from the response body (never the API key);
    ``retry_after`` is the suggested wait in seconds for rate limits.
    """

    def __init__(
        self,
        status_code: int,
        provider_message: str = "",
        retry_after: Optional[float] = None,
    ) -> None:
        self.status_code = status_code
        self.provider_message = provider_message
        self.retry_after = retry_after
        if status_code in (401, 403):
            self.kind = "auth"
            message = (
                f"Embedding API rejected the key (HTTP {status_code}) - "
                "check EMBEDDING_API_KEY / EMBEDDING_BASE_URL / EMBEDDING_PROVIDER"
            )
        elif status_code == 429:
            self.kind = "rate_limit"
            wait = f" - retry in {int(retry_after + 0.999)}s" if retry_after else " - retry shortly"
            message = f"Embedding rate limit reached (HTTP 429){wait}"
        elif status_code in (400, 404, 422):
            self.kind = "bad_request"
            message = f"Embedding API rejected the request (HTTP {status_code})"
        else:
            self.kind = "upstream"
            message = f"Embedding API error: HTTP {status_code}"
        if provider_message:
            message = f"{message}: {provider_message}"
        super().__init__(message, detail=provider_message or None)


class IngestionError(EWRGVError):
    """Raised when literature ingestion or document parsing fails."""


class RateLimitError(IngestionError):
    """A literature provider is rate limiting this client."""

    def __init__(self, message: str, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class RetrievalError(EWRGVError):
    """Raised when the retrieval layer cannot complete a query."""


class GapDetectionError(EWRGVError):
    """Raised when gap detection encounters an unrecoverable problem."""


class ValidationError(EWRGVError):
    """Raised when the EWRGV validation step fails."""


class StorageError(EWRGVError):
    """Raised when a storage backend (vector store, relational DB) fails."""


# ------------------------------------------------------------------ #
# HTTP-Aware API Exceptions
# ------------------------------------------------------------------ #


class APIError(EWRGVError):
    """
    Base for exceptions that carry an HTTP status code.

    The FastAPI exception handler converts these to structured JSON responses.
    """

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        detail: Optional[str] = None,
    ) -> None:
        super().__init__(message, detail)
        self.status_code = status_code


class NotFoundError(APIError):
    def __init__(self, message: str = "Resource not found") -> None:
        super().__init__(message, status_code=404)


class BadRequestError(APIError):
    def __init__(self, message: str = "Bad request") -> None:
        super().__init__(message, status_code=400)


class UnprocessableEntityError(APIError):
    def __init__(self, message: str = "Unprocessable entity") -> None:
        super().__init__(message, status_code=422)


class ServiceUnavailableError(APIError):
    def __init__(self, message: str = "Service temporarily unavailable") -> None:
        super().__init__(message, status_code=503)
