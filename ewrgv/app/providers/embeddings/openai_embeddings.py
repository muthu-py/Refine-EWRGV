"""
app/providers/embeddings/openai_embeddings.py
----------------------------------------------
OpenAI embedding provider implementation (Phase 2.9).

Provides async embedding via the OpenAI Embeddings API using httpx
(no openai SDK dependency — consistent with the rest of the codebase).

Features
--------
* Single text embedding via embed_text()
* Batch embedding via embed_documents()
* Sync wrappers embed() / embed_batch() for backward compatibility
* Retry with exponential backoff on transient failures (429, 500, 502, 503)
* Rate-limit awareness: honours Retry-After header
* Never logs API key
* Configurable model, dimension, timeout, retries

Security
--------
* API key is NEVER logged, repr'd, or included in exception messages.
* Only the last 4 characters are shown in debug messages for tracing.
"""

from __future__ import annotations

import asyncio
from typing import Optional

import httpx

from app.core.config import settings
from app.core.exceptions import ProviderError
from app.core.logging import get_logger

logger = get_logger(__name__)

# HTTP status codes that should trigger a retry
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503}

# OpenAI Embeddings API endpoint
_OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"


class OpenAIEmbeddingProvider:
    """
    EmbeddingProvider backed by OpenAI Embeddings API.

    Satisfies the EmbeddingProvider protocol from app.domain.interfaces.

    Parameters
    ----------
    api_key:
        OpenAI API key.  Defaults to settings.EMBEDDING_API_KEY.
    model:
        Embedding model name.  Defaults to settings.EMBEDDING_MODEL.
    dimension:
        Expected embedding dimension.  Defaults to settings.EMBEDDING_DIMENSION.
    timeout:
        HTTP request timeout in seconds.  Defaults to settings.EMBEDDING_TIMEOUT.
    max_retries:
        Maximum number of retries on transient failures.
    retry_delay:
        Base delay in seconds between retries (doubles each retry).
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimension: int | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
        retry_delay: float | None = None,
    ) -> None:
        self._api_key = api_key or settings.EMBEDDING_API_KEY
        self._model = model or settings.EMBEDDING_MODEL
        self._dimension = dimension or settings.EMBEDDING_DIMENSION
        self._timeout = timeout or settings.EMBEDDING_TIMEOUT
        self._max_retries = max_retries if max_retries is not None else settings.EMBEDDING_RETRY_COUNT
        self._retry_delay = retry_delay if retry_delay is not None else settings.EMBEDDING_RETRY_DELAY
        self._client: Optional[httpx.AsyncClient] = None

        if not self._api_key:
            logger.warning(
                "OpenAIEmbeddingProvider: No API key configured. "
                "Set EMBEDDING_API_KEY in .env."
            )

    # ------------------------------------------------------------------ #
    # Metadata accessors
    # ------------------------------------------------------------------ #

    def get_model_name(self) -> str:
        """Return the model identifier used by this provider."""
        return self._model

    def get_dimension(self) -> int:
        """Return the expected embedding vector dimension."""
        return self._dimension

    # ------------------------------------------------------------------ #
    # Async API (primary — Phase 2.9)
    # ------------------------------------------------------------------ #

    async def embed_text(self, text: str) -> list[float]:
        """
        Return a dense vector embedding for a single text string.

        This is also the query embedding interface:
            embed_text("online academic integrity and remote proctoring")
        """
        results = await self.embed_documents([text])
        return results[0]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Return embeddings for a batch of texts.

        Sends all texts in a single API request.  For large batches,
        the caller (RepresentationService) should split into configurable
        batch sizes before calling this method.

        Parameters
        ----------
        texts:
            List of text strings to embed.  Must not be empty.

        Returns
        -------
        list[list[float]]:
            One embedding vector per input text, in the same order.

        Raises
        ------
        ProviderError:
            On API failure, invalid response, or exhausted retries.
        """
        if not texts:
            return []

        if not self._api_key:
            raise ProviderError(
                "Embedding API key not configured. Set EMBEDDING_API_KEY."
            )

        client = await self._get_client()
        payload = {
            "input": texts,
            "model": self._model,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await client.post(
                    _OPENAI_EMBEDDINGS_URL,
                    json=payload,
                    headers=headers,
                    timeout=self._timeout,
                )

                if response.status_code == 200:
                    return self._parse_response(response.json(), len(texts))

                if response.status_code in _RETRYABLE_STATUS_CODES:
                    retry_after = self._get_retry_after(response, attempt)
                    logger.warning(
                        "Embedding API returned %d (attempt %d/%d), "
                        "retrying in %.1fs",
                        response.status_code,
                        attempt + 1,
                        self._max_retries + 1,
                        retry_after,
                    )
                    await asyncio.sleep(retry_after)
                    last_error = ProviderError(
                        f"Embedding API error: HTTP {response.status_code}",
                        detail=response.text[:500] if response.text else None,
                    )
                    continue

                # Non-retryable error
                raise ProviderError(
                    f"Embedding API error: HTTP {response.status_code}",
                    detail=response.text[:500] if response.text else None,
                )

            except httpx.RequestError as exc:
                logger.warning(
                    "Embedding API network error (attempt %d/%d): %s",
                    attempt + 1,
                    self._max_retries + 1,
                    str(exc),
                )
                last_error = ProviderError(
                    f"Embedding API network error: {exc}",
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(self._retry_delay * (2 ** attempt))
                    continue
                raise ProviderError(
                    f"Embedding API network error after {self._max_retries + 1} attempts",
                ) from exc

        # All retries exhausted
        raise last_error or ProviderError("Embedding API failed: retries exhausted")

    # ------------------------------------------------------------------ #
    # Sync API (backward compatibility)
    # ------------------------------------------------------------------ #

    def embed(self, text: str) -> list[float]:
        """Sync wrapper around embed_text() for backward compatibility."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            raise RuntimeError("embed() cannot be called from within a running asyncio loop. Use embed_text() instead.")
        return asyncio.run(self.embed_text(text))

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Sync wrapper around embed_documents() for backward compatibility."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            raise RuntimeError("embed_batch() cannot be called from within a running asyncio loop. Use embed_documents() instead.")
        return asyncio.run(self.embed_documents(texts))

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #

    async def close(self) -> None:
        """Release the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    async def _get_client(self) -> httpx.AsyncClient:
        """Lazy-initialize the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=float(self._timeout),
                follow_redirects=True,
            )
        return self._client

    def _parse_response(
        self, data: dict, expected_count: int
    ) -> list[list[float]]:
        """
        Parse the OpenAI embeddings response.

        Expected format:
            {"data": [{"embedding": [...], "index": 0}, ...], "model": "..."}
        """
        if "data" not in data:
            raise ProviderError(
                "Invalid embedding response: missing 'data' key",
                detail=str(data)[:200],
            )

        embeddings_data = data["data"]
        if len(embeddings_data) != expected_count:
            raise ProviderError(
                f"Embedding count mismatch: expected {expected_count}, "
                f"got {len(embeddings_data)}",
            )

        # Sort by index to ensure correct ordering
        sorted_data = sorted(embeddings_data, key=lambda x: x.get("index", 0))

        embeddings = []
        for item in sorted_data:
            vec = item.get("embedding", [])
            if not vec:
                raise ProviderError(
                    "Invalid embedding response: empty embedding vector",
                )
            # Verify dimension on first vector
            if embeddings == [] and len(vec) != self._dimension:
                logger.warning(
                    "Embedding dimension mismatch: expected %d, got %d. "
                    "Updating dimension to match actual provider response.",
                    self._dimension,
                    len(vec),
                )
                self._dimension = len(vec)
            embeddings.append(vec)

        return embeddings

    def _get_retry_after(self, response: httpx.Response, attempt: int) -> float:
        """
        Determine how long to wait before retrying.

        Uses Retry-After header if present, otherwise exponential backoff.
        """
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return float(retry_after)
            except (ValueError, TypeError):
                pass
        return self._retry_delay * (2 ** attempt)
