"""
app/retrieval/bm25.py
----------------------
Sparse BM25 keyword retrieval.

Self-contained BM25Okapi implementation — no external dependencies
(rank_bm25, Elasticsearch, etc.) are required.

BM25 Scoring Formula (Okapi BM25)
-----------------------------------
    score(q, d) = Σ IDF(t) × [ tf(t,d) × (k1 + 1) ]
                              / [ tf(t,d) + k1 × (1 - b + b × |d|/avgdl) ]

Where:
    - t: each query term
    - tf(t, d): term frequency of t in document d
    - |d|: length of document d (in tokens)
    - avgdl: average document length across the corpus
    - k1: term frequency saturation parameter (default 1.5)
    - b: length normalisation parameter (default 0.75)
    - IDF(t) = ln((N - n(t) + 0.5) / (n(t) + 0.5) + 1)
      where N = total documents, n(t) = documents containing term t

Flow:
    query → tokenize/normalize → BM25 score against all corpus chunks
    → return top-K results.
"""

from __future__ import annotations

import math
from collections import Counter

from app.core.logging import get_logger
from app.domain.interfaces import CorpusStore
from app.domain.models.paper import DocumentChunk
from app.domain.models.retrieval_result import RetrievalResult
from app.retrieval.interfaces import ScoredChunk
from app.stubs.stub_text_representation import StubLexicalRepresenter

logger = get_logger(__name__)


class SparseRetriever:
    """
    Retrieves document chunks using BM25 keyword matching.

    This is a self-contained BM25Okapi implementation that does not
    require any external library (rank_bm25, Elasticsearch, etc.).

    Parameters
    ----------
    corpus_store:
        Any object satisfying the CorpusStore protocol.
    tokenizer:
        A lexical tokenizer with a ``tokenize(text) -> list[str]`` method.
        Defaults to StubLexicalRepresenter.
    k1:
        BM25 term frequency saturation parameter. Default 1.5.
    b:
        BM25 document length normalisation parameter. Default 0.75.
    """

    def __init__(
        self,
        corpus_store: CorpusStore,
        tokenizer: StubLexicalRepresenter | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._corpus = corpus_store
        self._tokenizer = tokenizer or StubLexicalRepresenter()
        self._k1 = k1
        self._b = b

        # BM25 index state (built lazily)
        self._indexed = False
        self._chunks: list[DocumentChunk] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_freqs: dict[str, int] = {}  # term → number of docs containing it
        self._avgdl: float = 0.0
        self._n_docs: int = 0

    def _ensure_index(self) -> None:
        """Build the BM25 index from the corpus if not already built."""
        if self._indexed:
            return

        self._chunks = self._corpus.get_all_chunks()
        self._n_docs = len(self._chunks)

        if self._n_docs == 0:
            self._indexed = True
            return

        # Tokenize all documents
        self._doc_tokens = [
            self._tokenizer.tokenize(chunk.text) for chunk in self._chunks
        ]

        # Compute average document length
        total_tokens = sum(len(tokens) for tokens in self._doc_tokens)
        self._avgdl = total_tokens / self._n_docs if self._n_docs > 0 else 0.0

        # Compute document frequencies (how many docs contain each term)
        self._doc_freqs = {}
        for tokens in self._doc_tokens:
            unique_terms = set(tokens)
            for term in unique_terms:
                self._doc_freqs[term] = self._doc_freqs.get(term, 0) + 1

        self._indexed = True
        logger.info("SparseRetriever: indexed %d chunks", self._n_docs)

    def clear_index(self) -> None:
        """Clear the BM25 index to force recomputation."""
        self._indexed = False
        self._chunks = []
        self._doc_tokens = []
        self._doc_freqs = {}

    def _idf(self, term: str) -> float:
        """Compute IDF for a term using the BM25 IDF formula."""
        n_t = self._doc_freqs.get(term, 0)
        return math.log((self._n_docs - n_t + 0.5) / (n_t + 0.5) + 1.0)

    def _score_document(self, query_tokens: list[str], doc_idx: int) -> float:
        """Compute the BM25 score for a single document against query tokens."""
        doc_tokens = self._doc_tokens[doc_idx]
        doc_len = len(doc_tokens)

        if doc_len == 0:
            return 0.0

        # Term frequency in this document
        tf_counter = Counter(doc_tokens)

        score = 0.0
        for term in query_tokens:
            if term not in tf_counter:
                continue
            tf = tf_counter[term]
            idf = self._idf(term)

            # BM25 scoring formula
            numerator = tf * (self._k1 + 1.0)
            denominator = tf + self._k1 * (
                1.0 - self._b + self._b * doc_len / self._avgdl
            )
            score += idf * (numerator / denominator)

        return score

    def retrieve(self, query: str, top_k: int = 20) -> list[DocumentChunk]:
        """
        Satisfy the Retriever interface.

        Returns plain DocumentChunk objects without scores.
        """
        results = self.retrieve_results(query, top_k)
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
        Return BM25-scored chunks for internal fusion use.

        Satisfies the ScoredRetriever protocol (app.retrieval.interfaces).
        """
        self._ensure_index()

        if self._n_docs == 0:
            return []

        query_tokens = self._tokenizer.tokenize(query)
        if not query_tokens:
            return []

        # Score all documents
        scored: list[tuple[int, float]] = []
        for doc_idx in range(self._n_docs):
            score = self._score_document(query_tokens, doc_idx)
            if score > 0:
                scored.append((doc_idx, score))

        # Sort by score descending, then by chunk_id for determinism
        scored.sort(key=lambda x: (-x[1], self._chunks[x[0]].chunk_id))

        # Top-K
        top = scored[:top_k]

        return [
            ScoredChunk(
                chunk=self._chunks[doc_idx],
                score=score,
                retriever="sparse",
                rank=rank + 1,
            )
            for rank, (doc_idx, score) in enumerate(top)
        ]

    def retrieve_results(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        """
        Retrieve the top-K BM25-ranked document chunks as RetrievalResult objects.

        Parameters
        ----------
        query:
            The search query string.
        top_k:
            Number of top results to return.

        Returns
        -------
        list[RetrievalResult]
            Ranked list of retrieval results with ``retrieval_method="sparse"``.
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
                retrieval_method="sparse",
            )
            for sc in scored
        ]
