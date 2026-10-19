"""
app/providers/embeddings/sentence_transformer_embeddings.py
-------------------------------------------------------------
Local embedding provider backed by sentence-transformers (free, offline).

* Model loads lazily on first use and is cached per model name, so every
  request after the first reuses the same weights.
* Vectors are L2-normalised (cosine similarity == dot product).
* Async methods run the CPU-bound encode in a worker thread so the event
  loop stays responsive.
* No API key, no network after the first model download (Hugging Face cache).
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from app.core.config import settings
from app.core.exceptions import ProviderError
from app.core.logging import get_logger

logger = get_logger(__name__)

_MODELS: dict[str, Any] = {}
_MODELS_LOCK = threading.Lock()


def _load_model(model_name: str) -> Any:
    with _MODELS_LOCK:
        if model_name not in _MODELS:
            try:
                from sentence_transformers import SentenceTransformer  # noqa: PLC0415
            except ImportError as exc:
                raise ProviderError(
                    "sentence-transformers is not installed; run "
                    "`pip install sentence-transformers` or set EMBEDDING_PROVIDER=openai."
                ) from exc
            logger.info("Loading local embedding model %s", model_name)
            try:
                _MODELS[model_name] = SentenceTransformer(model_name)
            except Exception as exc:
                raise ProviderError(f"Could not load embedding model '{model_name}': {exc}") from exc
        return _MODELS[model_name]


class SentenceTransformerEmbeddingProvider:
    """
    EmbeddingProvider backed by a local sentence-transformers model.

    Satisfies the EmbeddingProvider protocol from app.domain.interfaces.
    """

    def __init__(
        self,
        model: str | None = None,
        dimension: int | None = None,
        batch_size: int | None = None,
    ) -> None:
        self._model = model or settings.EMBEDDING_MODEL
        self._dimension = dimension or settings.EMBEDDING_DIMENSION
        self._batch_size = max(1, batch_size or settings.EMBEDDING_BATCH_SIZE)

    # ------------------------------------------------------------------ #
    # Metadata accessors
    # ------------------------------------------------------------------ #

    def get_model_name(self) -> str:
        return self._model

    def get_dimension(self) -> int:
        return self._dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    # ------------------------------------------------------------------ #
    # Sync API
    # ------------------------------------------------------------------ #

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = _load_model(self._model)
        vectors = model.encode(
            list(texts),
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        result = [[float(x) for x in v] for v in vectors]
        actual = len(result[0])
        if actual != self._dimension:
            raise ProviderError(
                f"Model '{self._model}' returns {actual}-dim vectors but "
                f"EMBEDDING_DIMENSION={self._dimension}; update .env (and the "
                "chunk_embeddings column) to match."
            )
        return result

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    # ------------------------------------------------------------------ #
    # Async API
    # ------------------------------------------------------------------ #

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_batch, texts)

    async def embed_text(self, text: str) -> list[float]:
        return (await self.embed_documents([text]))[0]

    async def close(self) -> None:
        """Nothing to release; the cached model is shared across requests."""
