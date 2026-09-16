"""
app/retrieval/reranking.py
---------------------------
Cross-encoder reranking of retrieved document chunks.

The reranker refines the ranked list returned by the HybridRetriever by
applying a more expensive but more accurate relevance model.

Candidate implementations:
    - CrossEncoderReranker  (sentence-transformers cross-encoder models)
    - LLMReranker           (LLM-based relevance judgement – optional, expensive)

Status: SKELETON – interface is correct, implementation pending.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.models.paper import DocumentChunk

logger = get_logger(__name__)


class CrossEncoderReranker:
    """
    Reranks document chunks using a cross-encoder model.

    Satisfies the Reranker interface (app.domain.interfaces).
    Status: SKELETON – not implemented.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name
        self._model = None   # Will be a CrossEncoder instance

    def rerank(
        self,
        query: str,
        chunks: list[DocumentChunk],
        top_n: int = 10,
    ) -> list[DocumentChunk]:
        """
        Rerank chunks by relevance to the query.
        NOT YET IMPLEMENTED — returns input list truncated to top_n.
        """
        logger.warning("CrossEncoderReranker.rerank is not yet implemented.")
        return chunks[:top_n]
