"""
app/retrieval/service.py
--------------------------
Retrieval service — single entry point for all retrieval operations.

This service encapsulates retriever construction and method dispatch,
keeping route handlers thin and business logic out of the API layer
(per project convention).

The service is built over one research run's corpus (RunCorpus: the run's
PaperChunks, stored embeddings and BM25 index).  Dense retrieval needs an
embedding provider for the query; without one only ``sparse`` is offered
and ``hybrid`` falls back to sparse alone.

Use ``aretrieve`` from async code (FastAPI routes): it embeds the query —
and any chunk without a stored embedding — with the provider's async API
first, because the provider's sync ``embed`` cannot run inside an event
loop.
"""

from __future__ import annotations

from typing import Optional

from app.core.config import settings
from app.core.exceptions import BadRequestError, EmbeddingAPIError, RetrievalError
from app.core.logging import get_logger
from app.domain.interfaces import CorpusStore, EmbeddingProvider
from app.domain.models.retrieval_result import RetrievalResult
from app.retrieval.bm25 import SparseRetriever
from app.retrieval.filters import RetrievalFilters
from app.retrieval.fusion import HybridRetriever
from app.retrieval.semantic import DenseRetriever

logger = get_logger(__name__)

# Default: drop references / acknowledgments / boilerplate.
DEFAULT_FILTERS = RetrievalFilters()


class _PrecomputedEmbeddings:
    """EmbeddingProvider view that answers from vectors computed up front."""

    def __init__(self, provider: EmbeddingProvider, vectors: dict[str, list[float]]) -> None:
        self._provider = provider
        self._vectors = vectors

    def embed(self, text: str) -> list[float]:
        vector = self._vectors.get(text)
        return vector if vector is not None else self._provider.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if all(t in self._vectors for t in texts):
            return [self._vectors[t] for t in texts]
        return self._provider.embed_batch(texts)

    def get_model_name(self) -> str:
        return self._provider.get_model_name()

    def get_dimension(self) -> int:
        return self._provider.get_dimension()

# Valid retrieval method names.
VALID_METHODS = frozenset({"dense", "sparse", "hybrid"})


class RetrievalService:
    """
    Unified retrieval service dispatching to Dense, Sparse, or Hybrid.

    Parameters
    ----------
    corpus_store:
        The run's corpus (normally a RunCorpus from RunCorpusLoader).
    embedding_provider:
        Provider used to embed queries for dense retrieval.  Optional.
    """

    def __init__(
        self,
        corpus_store: CorpusStore,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> None:
        self._corpus = corpus_store
        self._embedder = embedding_provider
        self._sparse = SparseRetriever(corpus_store=corpus_store)
        self._dense: Optional[DenseRetriever] = None
        self._hybrid: Optional[HybridRetriever] = None
        if embedding_provider is not None:
            self._use_embedder(embedding_provider)

    def _use_embedder(self, embedder: EmbeddingProvider) -> None:
        self._dense = DenseRetriever(embedding_provider=embedder, corpus_store=self._corpus)
        self._hybrid = HybridRetriever(dense_retriever=self._dense, sparse_retriever=self._sparse)

    @property
    def corpus(self) -> CorpusStore:
        return self._corpus

    async def _prepare_dense(self, queries: list[str]) -> None:
        """
        Embed the queries (and chunks lacking stored vectors) with
        ``embed_documents`` so the synchronous retrievers never call the
        provider from inside the event loop.
        """
        queries = [q for q in queries if q and q.strip()]
        if self._embedder is None or not queries:
            return
        stored = getattr(self._corpus, "get_embedding", None)
        texts = queries + [
            c.text
            for c in self._corpus.get_all_chunks()
            if c.text and c.text.strip()
            and (not callable(stored) or stored(c.chunk_id) is None)
        ]
        texts = list(dict.fromkeys(texts))
        # Embed in batches: some providers cap inputs per request (Gemini: 100).
        batch_size = max(1, settings.EMBEDDING_BATCH_SIZE)
        try:
            vectors = []
            for i in range(0, len(texts), batch_size):
                vectors.extend(await self._embedder.embed_documents(texts[i:i + batch_size]))
        except EmbeddingAPIError:
            raise  # carries its own HTTP mapping (see app.main)
        except Exception as exc:
            raise RetrievalError(f"Query embedding failed: {exc}") from exc
        self._use_embedder(_PrecomputedEmbeddings(self._embedder, dict(zip(texts, vectors))))

    async def aretrieve(
        self,
        query: str,
        top_k: int = 10,
        method: str = "hybrid",
        filters: Optional[RetrievalFilters] = None,
    ) -> list[RetrievalResult]:
        """Async variant of ``retrieve`` for use inside an event loop."""
        if method in ("dense", "hybrid"):
            await self._prepare_dense([query])
        return self.retrieve(query, top_k, method, filters)

    async def aretrieve_many(
        self,
        queries: list[str],
        top_k: int = 10,
        method: str = "hybrid",
        filters: Optional[RetrievalFilters] = None,
        rrf_k: int = 60,
    ) -> list[RetrievalResult]:
        """
        Retrieve for several queries (e.g. the question plus its expansions)
        and fuse the ranked lists with Reciprocal Rank Fusion.

        Each result's metadata gains ``matched_queries`` (the queries that
        retrieved it) and ``query_ranks``.  The per-paper cap applies to the
        fused list.  Scores are fused RRF scores.
        """
        queries = list(dict.fromkeys(q.strip() for q in queries if q and q.strip()))
        if not queries:
            raise BadRequestError("At least one non-empty query is required.")
        if method in ("dense", "hybrid"):
            await self._prepare_dense(queries)

        fused: dict[str, float] = {}
        best: dict[str, RetrievalResult] = {}
        matched: dict[str, dict[str, int]] = {}
        for q in queries:
            ranked = self._ranked(q, top_k, method, filters)
            for rank, r in enumerate(ranked, start=1):
                fused[r.chunk_id] = fused.get(r.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
                best.setdefault(r.chunk_id, r)
                matched.setdefault(r.chunk_id, {})[q] = rank

        order = sorted(fused, key=lambda cid: fused[cid], reverse=True)
        results = [
            best[cid].model_copy(update={
                "score": fused[cid],
                "retrieval_method": f"multi_{method}",
                "metadata": {
                    **(best[cid].metadata or {}),
                    "matched_queries": list(matched[cid]),
                    "query_ranks": matched[cid],
                },
            })
            for cid in order
        ]
        return _cap_per_paper(results, top_k, settings.RETRIEVAL_MAX_PER_PAPER)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        method: str = "hybrid",
        filters: Optional[RetrievalFilters] = None,
    ) -> list[RetrievalResult]:
        """
        Run retrieval using the specified method.

        Raises
        ------
        BadRequestError
            If query is empty, method is invalid or unavailable, or top_k is
            out of range (1–100).
        RetrievalError
            If the retrieval operation fails unexpectedly.

        ``filters`` defaults to excluding references / acknowledgments /
        boilerplate; pass ``NO_FILTERS`` to search every chunk.
        """
        results = self._ranked(query, top_k, method, filters)
        return _cap_per_paper(results, top_k, settings.RETRIEVAL_MAX_PER_PAPER)

    def _ranked(
        self,
        query: str,
        top_k: int,
        method: str,
        filters: Optional[RetrievalFilters],
    ) -> list[RetrievalResult]:
        """Validated, filtered ranking (no per-paper cap), long enough to fill top_k."""
        if not query or not query.strip():
            raise BadRequestError("Query must be a non-empty string.")

        if method not in VALID_METHODS:
            raise BadRequestError(
                f"Invalid retrieval method '{method}'. "
                f"Supported methods: {', '.join(sorted(VALID_METHODS))}"
            )

        if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
            raise BadRequestError("top_k must be an integer between 1 and 100.")

        if method == "dense" and self._dense is None:
            raise BadRequestError(
                "Dense retrieval needs an embedding provider; configure EMBEDDING_PROVIDER (and EMBEDDING_API_KEY for API providers)."
            )

        filters = DEFAULT_FILTERS if filters is None else filters
        cap = settings.RETRIEVAL_MAX_PER_PAPER
        if filters.active:
            # Rank the whole corpus so filtering never starves the list
            # (a run's corpus is ~1k chunks; scoring all of them is cheap).
            pool = max(top_k, len(self._corpus.get_all_chunks()))
        else:
            # Over-fetch so the per-paper cap still leaves top_k results.
            pool = top_k * 10 if cap > 0 else top_k
        try:
            if method == "dense":
                results = self._dense.retrieve_results(query, pool)
            elif method == "sparse" or self._hybrid is None:
                results = self._sparse.retrieve_results(query, pool)
            else:
                results = self._hybrid.retrieve(query, pool)
        except Exception as exc:
            logger.error(
                "RetrievalService: retrieval failed",
                extra={"method": method, "query": query[:80], "error": str(exc)},
            )
            raise RetrievalError(f"Retrieval failed for method '{method}': {exc}") from exc
        return filters.apply(results)


def _cap_per_paper(results: list[RetrievalResult], top_k: int, cap: int) -> list[RetrievalResult]:
    """
    Keep at most ``cap`` results per paper so one paper cannot fill the list,
    preserving score order.  When fewer than ``top_k`` papers' worth of results
    exist, the best skipped results backfill the list.  Ranks are renumbered.
    """
    if cap <= 0:
        picked = results[:top_k]
    else:
        per_paper: dict[str, int] = {}
        picked, skipped = [], []
        for r in results:
            if per_paper.get(r.paper_id, 0) < cap:
                per_paper[r.paper_id] = per_paper.get(r.paper_id, 0) + 1
                picked.append(r)
            else:
                skipped.append(r)
            if len(picked) == top_k:
                break
        if len(picked) < top_k:
            picked += skipped[: top_k - len(picked)]
            picked.sort(key=lambda r: r.score, reverse=True)
    return [r.model_copy(update={"rank": i}) for i, r in enumerate(picked, start=1)]
