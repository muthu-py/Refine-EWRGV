"""
app/retrieval/service.py
--------------------------
Retrieval service — single entry point for all retrieval operations.

This service encapsulates retriever construction and method dispatch,
keeping route handlers thin and business logic out of the API layer
(per project convention).

When production implementations replace the stubs, this is the
**single point** where wiring changes — no API or retriever code
needs to be modified.
"""

from __future__ import annotations

from app.core.exceptions import BadRequestError, RetrievalError
from app.core.logging import get_logger
from app.domain.models.retrieval_result import RetrievalResult
from app.retrieval.bm25 import SparseRetriever
from app.retrieval.fusion import HybridRetriever
from app.retrieval.knowledge_graph import KnowledgeGraphRetriever
from app.retrieval.semantic import DenseRetriever
from app.stubs.stub_corpus import StubCorpusStore
from app.stubs.stub_knowledge_graph import StubKnowledgeGraphStore
from app.stubs.stub_text_representation import StubEmbeddingProvider

logger = get_logger(__name__)

# Valid retrieval method names.
VALID_METHODS = frozenset({"dense", "sparse", "knowledge_graph", "hybrid"})


class RetrievalService:
    """
    Unified retrieval service dispatching to Dense, Sparse, KG, or Hybrid.

    Construction wires up the stub dependencies (CorpusStore,
    EmbeddingProvider, KnowledgeGraphStore) and builds all four
    retrievers once.  The ``retrieve()`` method dispatches to the
    correct retriever based on the ``method`` parameter.

    Replace stubs here when production implementations are available —
    no API or retriever code changes needed.
    """

    def __init__(self) -> None:
        # --- Stub dependencies (swap these for production) ---
        self._corpus = StubCorpusStore()
        self._embedder = StubEmbeddingProvider()
        self._kg_store = StubKnowledgeGraphStore()

        # --- Retrievers ---
        self._dense = DenseRetriever(
            embedding_provider=self._embedder,
            corpus_store=self._corpus,
        )
        self._sparse = SparseRetriever(
            corpus_store=self._corpus,
        )
        self._kg = KnowledgeGraphRetriever(
            kg_store=self._kg_store,
            corpus_store=self._corpus,
        )
        self._hybrid = HybridRetriever(
            dense_retriever=self._dense,
            sparse_retriever=self._sparse,
            kg_retriever=self._kg,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        method: str = "hybrid",
    ) -> list[RetrievalResult]:
        """
        Run retrieval using the specified method.

        Parameters
        ----------
        query:
            The search query string (must be non-empty).
        top_k:
            Number of results to return (must be 1–100).
        method:
            Retrieval method: ``"dense"``, ``"sparse"``,
            ``"knowledge_graph"``, or ``"hybrid"``.

        Returns
        -------
        list[RetrievalResult]
            Ranked retrieval results.

        Raises
        ------
        BadRequestError
            If query is empty, method is invalid, or top_k is out of range.
        RetrievalError
            If the retrieval operation fails unexpectedly.
        """
        # --- Validation ---
        if not query or not query.strip():
            raise BadRequestError("Query must be a non-empty string.")

        if method not in VALID_METHODS:
            raise BadRequestError(
                f"Invalid retrieval method '{method}'. "
                f"Supported methods: {', '.join(sorted(VALID_METHODS))}"
            )

        if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
            raise BadRequestError(
                "top_k must be an integer between 1 and 100."
            )

        # --- Dispatch ---
        try:
            if method == "dense":
                return self._dense.retrieve_results(query, top_k)
            elif method == "sparse":
                return self._sparse.retrieve_results(query, top_k)
            elif method == "knowledge_graph":
                return self._kg.retrieve_results(query, top_k)
            else:  # hybrid
                return self._hybrid.retrieve(query, top_k)
        except Exception as exc:
            logger.error(
                "RetrievalService: retrieval failed",
                extra={"method": method, "query": query[:80], "error": str(exc)},
            )
            raise RetrievalError(
                f"Retrieval failed for method '{method}': {exc}"
            ) from exc
