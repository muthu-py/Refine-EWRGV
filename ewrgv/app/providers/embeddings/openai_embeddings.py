"""
app/providers/embeddings/openai_embeddings.py
----------------------------------------------
OpenAI embedding provider implementation.

Implements the EmbeddingProvider interface.
Status: SKELETON – not yet implemented.
"""

from __future__ import annotations

from app.core.logging import get_logger

logger = get_logger(__name__)


class OpenAIEmbeddingProvider:
    """
    EmbeddingProvider backed by OpenAI Embeddings API.

    Status: SKELETON – not yet implemented.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
    ) -> None:
        self._api_key = api_key
        self._model = model
        # self._client = openai.OpenAI(api_key=api_key)

    def embed(self, text: str) -> list[float]:
        """NOT YET IMPLEMENTED – returns empty list."""
        logger.warning("OpenAIEmbeddingProvider.embed is not yet implemented.")
        return []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """NOT YET IMPLEMENTED – returns empty list."""
        logger.warning("OpenAIEmbeddingProvider.embed_batch is not yet implemented.")
        return [[] for _ in texts]
