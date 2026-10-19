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
import random
import threading
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from app.core.exceptions import IngestionError, RateLimitError
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
        max_retries: int = 2,
        retry_base_delay: float = 2.0,
        min_request_interval: float = 1.0,
        max_retry_delay: float = 60.0,
        cache_ttl: int = 900,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._default_headers = default_headers or {}
        self._max_retries = max_retries
        self._retry_base_delay = max(0.0, retry_base_delay)
        self._min_request_interval = max(0.0, min_request_interval)
        self._max_retry_delay = max(0.0, max_retry_delay)
        self._cache_ttl = max(0, cache_ttl)
        self._lock = threading.Lock()
        self._next_request_at = 0.0
        self._cooldown_until = 0.0
        self._cache: dict[tuple[str, tuple[tuple[str, str], ...]], tuple[float, Any]] = {}
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
        cache_key = (url, tuple(sorted((str(k), str(v)) for k, v in (params or {}).items())))
        if self._cache_ttl:
            with self._lock:
                cached = self._cache.get(cache_key)
                if cached and cached[0] > time.monotonic():
                    logger.debug("LiteratureHttpClient: cache hit", extra={"url": url})
                    return cached[1]
                self._cache.pop(cache_key, None)
        attempt = 0
        last_error: Exception | None = None
        last_retry_after: float | None = None

        while attempt <= self._max_retries:
            try:
                with self._lock:
                    now = time.monotonic()
                    wait = max(self._cooldown_until, self._next_request_at) - now
                    if wait > 0:
                        time.sleep(wait)
                    self._next_request_at = time.monotonic() + self._min_request_interval
                response = self._client.get(url, params=params)
                if response.status_code == 200:
                    body = response.json()
                    if self._cache_ttl:
                        with self._lock:
                            self._cache[cache_key] = (time.monotonic() + self._cache_ttl, body)
                    return body
                elif response.status_code == 429:
                    retry_after = self._parse_retry_after(response.headers.get("Retry-After"))
                    wait = retry_after if retry_after is not None else min(
                        self._max_retry_delay,
                        self._retry_base_delay * (2 ** attempt) + random.uniform(0.0, 1.0),
                    )
                    last_retry_after = wait
                    with self._lock:
                        self._cooldown_until = max(self._cooldown_until, time.monotonic() + wait)
                    logger.warning(
                        "LiteratureHttpClient: rate limited (429)",
                        extra={"url": url, "attempt": attempt, "wait_s": wait},
                    )
                    if attempt < self._max_retries and wait <= self._max_retry_delay:
                        time.sleep(wait)
                    last_error = IngestionError(
                        f"Rate limited by {url} (HTTP 429)",
                        detail=response.text[:200],
                    )
                    if attempt >= self._max_retries or wait > self._max_retry_delay:
                        raise RateLimitError(
                            f"Rate limited by {url} (HTTP 429); provider cooldown is {wait:.0f}s",
                            retry_after=wait,
                        )
                elif response.status_code >= 500:
                    # Server error — backoff and retry
                    wait = min(self._max_retry_delay, self._retry_base_delay * (2 ** attempt) + random.uniform(0.0, 1.0))
                    logger.warning(
                        "LiteratureHttpClient: server error",
                        extra={
                            "url": url,
                            "status": response.status_code,
                            "attempt": attempt,
                            "wait_s": wait,
                        },
                    )
                    if attempt < self._max_retries:
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

        if last_retry_after is not None:
            raise RateLimitError(str(last_error), retry_after=last_retry_after)
        raise last_error or IngestionError(f"Failed to GET {url} after {self._max_retries} retries.")

    @staticmethod
    def _parse_retry_after(value: str | None) -> float | None:
        """Parse Retry-After as seconds or an HTTP date."""
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                return None

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
