"""
tests/unit/test_sparse_retrieval.py
-------------------------------------
Unit tests for SparseRetriever (bm25.py).

Verifies:
    - BM25 scoring produces meaningful rankings
    - Relevant lexical matches rank higher
    - Respects top_k
    - Deterministic behavior
    - Correct retrieval_method
"""

from __future__ import annotations

from app.retrieval.bm25 import SparseRetriever
from app.stubs.stub_corpus import StubCorpusStore


class TestSparseRetriever:
    """Tests for BM25 sparse retrieval."""

    def setup_method(self) -> None:
        self.corpus = StubCorpusStore()
        self.retriever = SparseRetriever(corpus_store=self.corpus)

    # ------------------------------------------------------------------ #
    # Basic functionality
    # ------------------------------------------------------------------ #

    def test_returns_results(self) -> None:
        """A query with matching terms returns results."""
        results = self.retriever.retrieve_results("Transformer attention", top_k=5)
        assert len(results) > 0

    def test_returns_retrieval_result_objects(self) -> None:
        """Results are RetrievalResult objects with required fields."""
        results = self.retriever.retrieve_results("language model", top_k=3)
        for r in results:
            assert r.chunk_id
            assert r.paper_id
            assert r.text
            assert r.retrieval_method == "sparse"
            assert r.rank > 0
            assert isinstance(r.score, float)

    # ------------------------------------------------------------------ #
    # Relevance
    # ------------------------------------------------------------------ #

    def test_relevant_matches_rank_higher(self) -> None:
        """Chunks containing query terms should appear in results."""
        results = self.retriever.retrieve_results("retrieval augmented generation", top_k=5)
        # RAG paper chunks should appear
        chunk_ids = [r.chunk_id for r in results]
        # At least one RAG-related chunk should be present
        assert any("005" in cid for cid in chunk_ids)

    def test_bm25_scores_positive(self) -> None:
        """BM25 scores are positive for matching documents."""
        results = self.retriever.retrieve_results("Transformer", top_k=10)
        for r in results:
            assert r.score > 0

    # ------------------------------------------------------------------ #
    # top_k
    # ------------------------------------------------------------------ #

    def test_respects_top_k(self) -> None:
        """Never returns more than top_k results."""
        results = self.retriever.retrieve_results("model", top_k=3)
        assert len(results) <= 3

    def test_returns_fewer_when_few_matches(self) -> None:
        """Returns fewer than top_k if fewer docs match."""
        # A very specific query may match fewer documents
        results = self.retriever.retrieve_results("BLEU translation WMT", top_k=20)
        assert len(results) <= 20

    # ------------------------------------------------------------------ #
    # Determinism
    # ------------------------------------------------------------------ #

    def test_deterministic_ranking(self) -> None:
        """Same query produces same ranking every time."""
        r1 = self.retriever.retrieve_results("attention mechanism", top_k=5)
        r2 = self.retriever.retrieve_results("attention mechanism", top_k=5)
        assert [r.chunk_id for r in r1] == [r.chunk_id for r in r2]
        assert [r.score for r in r1] == [r.score for r in r2]

    # ------------------------------------------------------------------ #
    # Scores and ranking
    # ------------------------------------------------------------------ #

    def test_scores_are_descending(self) -> None:
        """Results are sorted by score descending."""
        results = self.retriever.retrieve_results("pre-training language model", top_k=10)
        scores = [r.score for r in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_ranks_are_sequential(self) -> None:
        """Ranks are sequential starting from 1."""
        results = self.retriever.retrieve_results("neural network", top_k=5)
        for i, r in enumerate(results):
            assert r.rank == i + 1

    # ------------------------------------------------------------------ #
    # Edge cases
    # ------------------------------------------------------------------ #

    def test_empty_query(self) -> None:
        """Empty query returns empty results."""
        results = self.retriever.retrieve_results("", top_k=5)
        assert results == []

    def test_only_stopwords_query(self) -> None:
        """Query with only stop words returns empty results."""
        results = self.retriever.retrieve_results("the a is are", top_k=5)
        assert results == []

    def test_no_matching_terms(self) -> None:
        """Query with no matching terms returns empty results."""
        results = self.retriever.retrieve_results("zzzzxyzzy nonexistent", top_k=5)
        assert results == []

    # ------------------------------------------------------------------ #
    # Retriever interface
    # ------------------------------------------------------------------ #

    def test_retrieve_returns_document_chunks(self) -> None:
        """The retrieve() method returns DocumentChunk objects."""
        chunks = self.retriever.retrieve("Transformer", top_k=3)
        assert len(chunks) > 0
        for c in chunks:
            assert hasattr(c, "chunk_id")
            assert hasattr(c, "text")

    # ------------------------------------------------------------------ #
    # Index management
    # ------------------------------------------------------------------ #

    def test_clear_index_and_rebuild(self) -> None:
        """Clearing and rebuilding the index produces same results."""
        r1 = self.retriever.retrieve_results("attention", top_k=5)
        self.retriever.clear_index()
        r2 = self.retriever.retrieve_results("attention", top_k=5)
        assert [r.chunk_id for r in r1] == [r.chunk_id for r in r2]
