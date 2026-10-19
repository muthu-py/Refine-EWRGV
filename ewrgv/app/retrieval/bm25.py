"""
app/retrieval/bm25.py
----------------------
Sparse BM25 keyword retrieval over a run's PaperChunks.

Scoring is delegated to BM25Index (app.retrieval.bm25_index, rank-bm25
BM25Okapi with k1=1.5, b=0.75), the same index that RepresentationService
builds and persists in Phase 2.9.  When the corpus carries a loaded index
(RunCorpus.bm25_index) it is used as-is; otherwise one is built from the
corpus chunks on first use.

Flow:
    query → preprocess_text → BM25Okapi scores → top-K PaperChunks.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.interfaces import CorpusStore
from app.domain.models.paper_chunk import PaperChunk
from app.domain.models.retrieval_result import RetrievalResult
from app.retrieval.bm25_index import BM25Index
from app.retrieval.interfaces import ScoredChunk, to_retrieval_result

logger = get_logger(__name__)


class SparseRetriever:
    """
    Retrieves PaperChunks using BM25 keyword matching.

    Parameters
    ----------
    corpus_store:
        Any object satisfying the CorpusStore protocol.  If it exposes a
        ``bm25_index`` attribute (RunCorpus does), that index is used.
    bm25_index:
        Explicit index to use instead of the corpus's.
    """

    def __init__(
        self,
        corpus_store: CorpusStore,
        bm25_index: BM25Index | None = None,
    ) -> None:
        self._corpus = corpus_store
        self._index = bm25_index

    def _ensure_index(self) -> BM25Index:
        if self._index is None:
            corpus_index = getattr(self._corpus, "bm25_index", None)
            self._index = (
                corpus_index
                if isinstance(corpus_index, BM25Index)
                else BM25Index().build(self._corpus.get_all_chunks())
            )
            logger.info("SparseRetriever: using BM25 index over %d chunks", self._index.size)
        return self._index

    def clear_index(self) -> None:
        """Drop the cached index so the next query rebuilds/reloads it."""
        self._index = None

    def retrieve(self, query: str, top_k: int = 20) -> list[PaperChunk]:
        """Satisfy the Retriever interface: ranked PaperChunks without scores."""
        return [sc.chunk for sc in self.retrieve_scored(query, top_k)]

    def retrieve_scored(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        """
        Return BM25-scored chunks for internal fusion use.

        Ties are broken by chunk_id so the order is deterministic.
        Index entries whose chunk is no longer in the corpus are skipped.
        """
        index = self._ensure_index()
        if index.is_empty() or not query or not query.strip():
            return []

        hits = index.query(query, top_k=index.size)
        hits.sort(key=lambda hit: (-hit[1], hit[0]))

        scored: list[ScoredChunk] = []
        for chunk_id, score in hits:
            chunk = self._corpus.get_chunk_by_id(chunk_id)
            if chunk is None:
                continue
            scored.append(
                ScoredChunk(chunk=chunk, score=float(score), retriever="sparse", rank=len(scored) + 1)
            )
            if len(scored) >= top_k:
                break
        return scored

    def retrieve_results(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        """Top-K BM25 results as RetrievalResult objects (``retrieval_method="sparse"``)."""
        return [
            to_retrieval_result(sc, "sparse", self._corpus.get_paper_by_id(sc.chunk.paper_id))
            for sc in self.retrieve_scored(query, top_k)
        ]
