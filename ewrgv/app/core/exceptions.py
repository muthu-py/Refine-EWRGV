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


class IngestionError(EWRGVError):
    """Raised when literature ingestion or document parsing fails."""


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
