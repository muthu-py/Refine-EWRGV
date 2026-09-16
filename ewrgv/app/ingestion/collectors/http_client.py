"""
app/ingestion/collectors/http_client.py
----------------------------------------
Reusable HTTP client for literature provider adapters.

Design notes
------------
* Built on ``httpx`` (already in requirements.txt) for connection reuse.
* Supports configurable base URL, timeout, and custom default headers.
* Bounded retry logic: up to ``max_retries`` attempts with exponential
  backoff for 429 / 5xx responses.  Permanent 4xx errors (except 429) are
  NOT retried.
* All errors surface as ``IngestionError`` so callers depend only on the
  EWRGV exception hierarchy, never on httpx internals.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from app.core.exceptions import IngestionError
from app.core.logging import get_logger

logger = get_logger(__name__)


class LiteratureHttpClient:
    """
    Thin, reusable HTTP client for literature provider adapters.

    Parameters
    ----------
    base_url:
        Base URL prepended to every request path.
    timeout:
        Request timeout in seconds.
    default_headers:
        Headers added to every request (e.g. User-Agent, API key).
    max_retries:
        Maximum number of retry attempts for transient failures (429 / 5xx).
        Set to 0 to disable retries.
    """

    def __init__(
        self,
        base_url: str = "",
        timeout: int = 30,
        default_headers: dict[str, str] | None = None,
        max_retries: int = 3,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._default_headers = default_headers or {}
        self._max_retries = max_retries
        self._client = httpx.Client(
            timeout=timeout,
            headers=self._default_headers,
        )

    # ------------------------------------------------------------------ #
    # Public interface
    # ------------------------------------------------------------------ #

    def get(self, path: str, params: dict[str, Any] | None = None) -> dict | list:
        """
        Perform a GET request and return the parsed JSON body.

        Parameters
        ----------
        path:
            URL path appended to ``base_url``.
        params:
            Query string parameters.

        Returns
        -------
        dict | list
            Parsed JSON response body.

        Raises
        ------
        IngestionError
            On HTTP errors, timeout, or unparseable JSON.
        """
        url = f"{self._base_url}/{path.lstrip('/')}" if self._base_url else path
        attempt = 0
        last_error: Exception | None = None

        while attempt <= self._max_retries:
            try:
                response = self._client.get(url, params=params)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Rate limited — backoff and retry
                    wait = 2 ** attempt
                    logger.warning(
                        "LiteratureHttpClient: rate limited (429)",
                        extra={"url": url, "attempt": attempt, "wait_s": wait},
                    )
                    time.sleep(wait)
                    last_error = IngestionError(
                        f"Rate limited by {url} (HTTP 429)",
                        detail=response.text[:200],
                    )
                elif response.status_code >= 500:
                    # Server error — backoff and retry
                    wait = 2 ** attempt
                    logger.warning(
                        "LiteratureHttpClient: server error",
                        extra={
                            "url": url,
                            "status": response.status_code,
                            "attempt": attempt,
                            "wait_s": wait,
                        },
                    )
                    time.sleep(wait)
                    last_error = IngestionError(
                        f"Server error from {url} (HTTP {response.status_code})",
                        detail=response.text[:200],
                    )
                else:
                    # Permanent client error (400, 401, 403, 404 …) — no retry
                    raise IngestionError(
                        f"HTTP {response.status_code} from {url}",
                        detail=response.text[:200],
                    )
            except httpx.TimeoutException as exc:
                last_error = IngestionError(
                    f"Request to {url} timed out after {self._timeout}s",
                    detail=str(exc),
                )
                logger.warning(
                    "LiteratureHttpClient: timeout",
                    extra={"url": url, "attempt": attempt},
                )
            except httpx.RequestError as exc:
                raise IngestionError(
                    f"Network error reaching {url}: {exc}",
                    detail=str(exc),
                ) from exc
            except IngestionError:
                raise  # permanent errors — don't retry

            attempt += 1

        raise last_error or IngestionError(f"Failed to GET {url} after {self._max_retries} retries.")

    def close(self) -> None:
        """Release underlying httpx connection pool."""
        self._client.close()

    # ------------------------------------------------------------------ #
    # Context-manager support
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "LiteratureHttpClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
