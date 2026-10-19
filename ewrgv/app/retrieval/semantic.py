"""
app/retrieval/semantic.py
--------------------------
Dense semantic retrieval using vector similarity search.

Depends on:
    EmbeddingProvider  (app.domain.interfaces) — for query/chunk embedding
    CorpusStore        (app.domain.interfaces) — for accessing document chunks

Flow:
    query → embed query → compute cosine similarity against all corpus
    chunk embeddings → return top-K results sorted by similarity score.

Similarity metric: Cosine Similarity
    cos(a, b) = (a · b) / (‖a‖ × ‖b‖)
    Range: [-1, 1] for general vectors; [0, 1] for normalised vectors.
"""

from __future__ import annotations

import math

from app.core.logging import get_logger
from app.domain.interfaces import CorpusStore, EmbeddingProvider
from app.domain.models.paper import DocumentChunk
from app.domain.models.retrieval_result import RetrievalResult
from app.retrieval.interfaces import ScoredChunk

logger = get_logger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Returns a float in [-1.0, 1.0].  For unit-normalised vectors this
    simplifies to the dot product.
    """
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class DenseRetriever:
    """
    Retrieves document chunks using dense vector similarity search.

    Implementation
    --------------
    1. Embed the query string using the injected EmbeddingProvider.
    2. Embed all corpus chunks (cached after first call).
    3. Compute cosine similarity between the query embedding and each
       chunk embedding.
    4. Return the top-K most similar chunks as RetrievalResult objects.

    Parameters
    ----------
    embedding_provider:
        Any object satisfying the EmbeddingProvider protocol.
    corpus_store:
        Any object satisfying the CorpusStore protocol.

    Notes
    -----
    Chunk embeddings are computed lazily on the first retrieval call and
    cached in memory.  Call ``clear_cache()`` to force recomputation.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        corpus_store: CorpusStore,
    ) -> None:
        self._embedder = embedding_provider
        self._corpus = corpus_store
        self._chunk_embeddings: list[tuple[DocumentChunk, list[float]]] | None = None

    def _ensure_index(self) -> list[tuple[DocumentChunk, list[float]]]:
        """Build or return cached chunk embeddings."""
        if self._chunk_embeddings is None:
            chunks = self._corpus.get_all_chunks()
            texts = [c.text for c in chunks]
            embeddings = self._embedder.embed_batch(texts)
            self._chunk_embeddings = list(zip(chunks, embeddings))
            logger.info(
                "DenseRetriever: indexed %d chunks", len(self._chunk_embeddings)
            )
        return self._chunk_embeddings

    def clear_cache(self) -> None:
        """Clear the cached chunk embeddings to force recomputation."""
        self._chunk_embeddings = None

    def retrieve(self, query: str, top_k: int = 20) -> list[DocumentChunk]:
        """
        Satisfy the Retriever interface.

        Returns plain DocumentChunk objects without scores.
        """
        results = self.retrieve_results(query, top_k)
        # Reconstruct DocumentChunk from result fields
        return [
            DocumentChunk(
                chunk_id=r.chunk_id,
                paper_id=r.paper_id,
                section=r.section,
                text=r.text,
            )
            for r in results
        ]

    def retrieve_scored(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        """
        Return scored chunks for internal fusion use.

        Satisfies the ScoredRetriever protocol (app.retrieval.interfaces).
        """
        index = self._ensure_index()
        query_embedding = self._embedder.embed(query)

        # Score all chunks
        scored: list[tuple[DocumentChunk, float]] = []
        for chunk, chunk_emb in index:
            sim = _cosine_similarity(query_embedding, chunk_emb)
            scored.append((chunk, sim))

        # Sort by similarity descending, then by chunk_id for determinism
        scored.sort(key=lambda x: (-x[1], x[0].chunk_id))

        # Top-K
        top = scored[:top_k]

        return [
            ScoredChunk(
                chunk=chunk,
                score=score,
                retriever="dense",
                rank=rank + 1,
            )
            for rank, (chunk, score) in enumerate(top)
        ]

    def retrieve_results(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        """
        Retrieve the top-K most similar document chunks as RetrievalResult objects.

        Parameters
        ----------
        query:
            The search query string.
        top_k:
            Number of top results to return.

        Returns
        -------
        list[RetrievalResult]
            Ranked list of retrieval results with ``retrieval_method="dense"``.
        """
        scored = self.retrieve_scored(query, top_k)

        return [
            RetrievalResult(
                chunk_id=sc.chunk.chunk_id,
                paper_id=sc.chunk.paper_id,
                section=sc.chunk.section,
                text=sc.chunk.text,
                score=sc.score,
                rank=sc.rank,
                retrieval_method="dense",
            )
            for sc in scored
        ]
