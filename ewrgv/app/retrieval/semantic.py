"""
app/retrieval/semantic.py
--------------------------
Dense semantic retrieval using vector similarity search.

Depends on:
    EmbeddingProvider  (app.domain.interfaces)
    VectorStore        (app.domain.interfaces)

NOT YET IMPLEMENTED — placeholder class with correct interface contract.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.interfaces import EmbeddingProvider, VectorStore
from app.domain.models.paper import DocumentChunk
from app.retrieval.interfaces import ScoredChunk

logger = get_logger(__name__)


class SemanticRetriever:
    """
    Retrieves document chunks using dense vector similarity search.

    Implementation contract
    -----------------------
    1. Embed the query string using the injected EmbeddingProvider.
    2. Query the VectorStore for the top-k nearest neighbours.
    3. Return ScoredChunks for use by the fusion module.

    Status: SKELETON – not implemented.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self._embedder = embedding_provider
        self._store = vector_store

    def retrieve_scored(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        """
        Return semantically-similar chunks for the query.
        NOT YET IMPLEMENTED.
        """
        logger.warning("SemanticRetriever.retrieve_scored is not yet implemented.")
        return []

    def retrieve(self, query: str, top_k: int = 20) -> list[DocumentChunk]:
        """Satisfy the Retriever interface."""
        return [sc.chunk for sc in self.retrieve_scored(query, top_k)]
