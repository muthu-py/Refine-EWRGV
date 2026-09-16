"""
app/retrieval/bm25.py
----------------------
Sparse BM25 keyword retrieval.

Depends on an in-memory or persisted BM25 index built from DocumentChunks.
Candidate library: ``rank_bm25``.

Status: SKELETON – not implemented.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.models.paper import DocumentChunk
from app.retrieval.interfaces import ScoredChunk

logger = get_logger(__name__)


class BM25Retriever:
    """
    Retrieves document chunks using BM25 keyword matching.

    Status: SKELETON – not implemented.
    """

    def __init__(self) -> None:
        self._index = None   # Will be a rank_bm25.BM25Okapi or equivalent

    def build_index(self, chunks: list[DocumentChunk]) -> None:
        """Build or rebuild the BM25 index from the current corpus."""
        logger.warning("BM25Retriever.build_index is not yet implemented.")

    def retrieve_scored(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        """Return BM25-scored chunks. NOT YET IMPLEMENTED."""
        logger.warning("BM25Retriever.retrieve_scored is not yet implemented.")
        return []

    def retrieve(self, query: str, top_k: int = 20) -> list[DocumentChunk]:
        """Satisfy the Retriever interface."""
        return [sc.chunk for sc in self.retrieve_scored(query, top_k)]
