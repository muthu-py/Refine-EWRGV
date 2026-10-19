"""
app/retrieval/semantic.py
--------------------------
Dense semantic retrieval using vector similarity search.

Depends on:
    EmbeddingProvider  (app.domain.interfaces) — embeds the query
    CorpusStore        (app.domain.interfaces) — the run's PaperChunks

Chunk vectors come from the corpus when it carries stored embeddings
(RunCorpus.get_embedding, loaded from the ``chunk_embeddings`` table in
Phase 2.9).  Chunks without a stored vector are embedded once with the
provider and cached, so a partially embedded run still retrieves fully.

Similarity metric: Cosine Similarity
    cos(a, b) = (a · b) / (‖a‖ × ‖b‖)
"""

from __future__ import annotations

import math

from app.core.logging import get_logger
from app.domain.interfaces import CorpusStore, EmbeddingProvider
from app.domain.models.paper_chunk import PaperChunk
from app.domain.models.retrieval_result import RetrievalResult
from app.retrieval.interfaces import ScoredChunk, to_retrieval_result

logger = get_logger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity in [-1, 1]; 0.0 for mismatched or zero vectors."""
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
    Retrieves PaperChunks by cosine similarity to the query embedding.

    Parameters
    ----------
    embedding_provider:
        Any object satisfying the EmbeddingProvider protocol.  Must be the
        same model that produced the stored chunk embeddings.
    corpus_store:
        Any object satisfying the CorpusStore protocol.  If it exposes
        ``get_embedding(chunk_id)`` (RunCorpus does), stored vectors are used.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        corpus_store: CorpusStore,
    ) -> None:
        self._embedder = embedding_provider
        self._corpus = corpus_store
        self._chunk_embeddings: list[tuple[PaperChunk, list[float]]] | None = None

    def _ensure_index(self) -> list[tuple[PaperChunk, list[float]]]:
        """Collect stored vectors and embed any chunk that has none (cached)."""
        if self._chunk_embeddings is None:
            stored = getattr(self._corpus, "get_embedding", None)
            pairs: list[tuple[PaperChunk, list[float] | None]] = [
                (c, stored(c.chunk_id) if callable(stored) else None)
                for c in self._corpus.get_all_chunks()
            ]
            missing = [c for c, vec in pairs if vec is None]
            computed: dict[str, list[float]] = {}
            if missing:
                vectors = self._embedder.embed_batch([c.text for c in missing])
                computed = {c.chunk_id: v for c, v in zip(missing, vectors)}
            self._chunk_embeddings = [
                (c, vec if vec is not None else computed[c.chunk_id]) for c, vec in pairs
            ]
            logger.info(
                "DenseRetriever: indexed %d chunks (%d embedded on demand)",
                len(self._chunk_embeddings),
                len(missing),
            )
        return self._chunk_embeddings

    def clear_cache(self) -> None:
        """Clear the cached chunk embeddings to force recomputation."""
        self._chunk_embeddings = None

    def retrieve(self, query: str, top_k: int = 20) -> list[PaperChunk]:
        """Satisfy the Retriever interface: ranked PaperChunks without scores."""
        return [sc.chunk for sc in self.retrieve_scored(query, top_k)]

    def retrieve_scored(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        """Return cosine-scored chunks for internal fusion use (ties by chunk_id)."""
        index = self._ensure_index()
        if not index or not query or not query.strip():
            return []
        query_embedding = self._embedder.embed(query)

        scored = [(chunk, _cosine_similarity(query_embedding, emb)) for chunk, emb in index]
        scored.sort(key=lambda x: (-x[1], x[0].chunk_id))

        return [
            ScoredChunk(chunk=chunk, score=score, retriever="dense", rank=rank + 1)
            for rank, (chunk, score) in enumerate(scored[:top_k])
        ]

    def retrieve_results(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        """Top-K dense results as RetrievalResult objects (``retrieval_method="dense"``)."""
        return [
            to_retrieval_result(sc, "dense", self._corpus.get_paper_by_id(sc.chunk.paper_id))
            for sc in self.retrieve_scored(query, top_k)
        ]
