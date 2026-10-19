"""
tests/unit/test_dense_retrieval.py
------------------------------------
Unit tests for DenseRetriever (semantic.py).

Verifies:
    - Returns relevant deterministic results
    - Respects top_k
    - Ranking is deterministic
    - Correct retrieval_method
    - Scores are valid cosine similarity values
"""

from __future__ import annotations

from app.retrieval.semantic import DenseRetriever
from app.stubs.stub_corpus import StubCorpusStore
from app.stubs.stub_text_representation import StubEmbeddingProvider


class TestDenseRetriever:
    """Tests for dense/semantic retrieval."""

    def setup_method(self) -> None:
        self.corpus = StubCorpusStore()
        self.embedder = StubEmbeddingProvider()
        self.retriever = DenseRetriever(
            embedding_provider=self.embedder,
            corpus_store=self.corpus,
        )

    # ------------------------------------------------------------------ #
    # Basic functionality
    # ------------------------------------------------------------------ #

    def test_returns_results(self) -> None:
        """A query returns non-empty results."""
        results = self.retriever.retrieve_results("Transformer attention", top_k=5)
        assert len(results) > 0

    def test_returns_retrieval_result_objects(self) -> None:
        """Results are RetrievalResult objects with required fields."""
        results = self.retriever.retrieve_results("neural network", top_k=3)
        for r in results:
            assert r.chunk_id
            assert r.paper_id
            assert r.text
            assert r.retrieval_method == "dense"
            assert r.rank > 0
            assert isinstance(r.score, float)

    # ------------------------------------------------------------------ #
    # top_k
    # ------------------------------------------------------------------ #

    def test_respects_top_k(self) -> None:
        """Never returns more than top_k results."""
        results = self.retriever.retrieve_results("language model", top_k=3)
        assert len(results) <= 3

    def test_top_k_larger_than_corpus(self) -> None:
        """top_k larger than corpus returns all chunks."""
        results = self.retriever.retrieve_results("model", top_k=100)
        assert len(results) == 13  # Total chunks in mock corpus

    # ------------------------------------------------------------------ #
    # Determinism
    # ------------------------------------------------------------------ #

    def test_deterministic_ranking(self) -> None:
        """Same query produces same ranking every time."""
        r1 = self.retriever.retrieve_results("Transformer", top_k=5)
        r2 = self.retriever.retrieve_results("Transformer", top_k=5)
        assert [r.chunk_id for r in r1] == [r.chunk_id for r in r2]
        assert [r.score for r in r1] == [r.score for r in r2]

    # ------------------------------------------------------------------ #
    # Scores
    # ------------------------------------------------------------------ #

    def test_scores_are_valid(self) -> None:
        """Scores are within the cosine similarity range [-1, 1]."""
        results = self.retriever.retrieve_results("attention mechanism", top_k=5)
        for r in results:
            assert -1.0 <= r.score <= 1.0

    def test_scores_are_descending(self) -> None:
        """Results are sorted by score descending."""
        results = self.retriever.retrieve_results("BERT pre-training", top_k=10)
        scores = [r.score for r in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    # ------------------------------------------------------------------ #
    # Ranks
    # ------------------------------------------------------------------ #

    def test_ranks_are_sequential(self) -> None:
        """Ranks are sequential starting from 1."""
        results = self.retriever.retrieve_results("language model", top_k=5)
        for i, r in enumerate(results):
            assert r.rank == i + 1

    # ------------------------------------------------------------------ #
    # Retriever interface compatibility
    # ------------------------------------------------------------------ #

    def test_retrieve_returns_document_chunks(self) -> None:
        """The retrieve() method returns DocumentChunk objects."""
        chunks = self.retriever.retrieve("Transformer", top_k=3)
        assert len(chunks) > 0
        for c in chunks:
            assert hasattr(c, "chunk_id")
            assert hasattr(c, "paper_id")
            assert hasattr(c, "text")

    # ------------------------------------------------------------------ #
    # Cache
    # ------------------------------------------------------------------ #

    def test_clear_cache(self) -> None:
        """Clearing cache and re-running still produces same results."""
        r1 = self.retriever.retrieve_results("test", top_k=3)
        self.retriever.clear_cache()
        r2 = self.retriever.retrieve_results("test", top_k=3)
        assert [r.chunk_id for r in r1] == [r.chunk_id for r in r2]
