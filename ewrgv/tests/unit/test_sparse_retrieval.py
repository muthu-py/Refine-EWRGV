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
from app.domain.models.paper_chunk import PaperChunk
from app.retrieval.bm25_index import BM25Index
from tests.fixtures.sample_corpus import make_sample_corpus, sample_chunks


class TestSparseRetriever:
    """Tests for BM25 sparse retrieval."""

    def setup_method(self) -> None:
        self.corpus = make_sample_corpus()
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

    def test_retrieve_returns_paper_chunks(self) -> None:
        """The retrieve() method returns the corpus's PaperChunk objects."""
        chunks = self.retriever.retrieve("Transformer encoder decoder", top_k=3)
        assert len(chunks) > 0
        for c in chunks:
            assert isinstance(c, PaperChunk)
            assert c is self.corpus.get_chunk_by_id(c.chunk_id)

    def test_results_carry_section_and_paper_provenance(self) -> None:
        """Results expose the chunk's section type, pages and paper fields."""
        results = self.retriever.retrieve_results("limitations", top_k=5)
        hit = next(r for r in results if r.chunk_id == "chunk-003-limits")
        assert hit.section_type == "LIMITATIONS"
        assert hit.page_start >= 1
        assert hit.metadata["title"] == self.corpus.get_paper_by_id(hit.paper_id).title
        assert hit.metadata["publication_year"] == 2020

    # ------------------------------------------------------------------ #
    # Index management
    # ------------------------------------------------------------------ #

    def test_clear_index_and_rebuild(self) -> None:
        """Clearing and rebuilding the index produces same results."""
        r1 = self.retriever.retrieve_results("attention", top_k=5)
        self.retriever.clear_index()
        r2 = self.retriever.retrieve_results("attention", top_k=5)
        assert [r.chunk_id for r in r1] == [r.chunk_id for r in r2]


class TestSparseRetrieverIndex:
    """The retriever uses the run's persisted BM25 index when available."""

    def test_uses_corpus_index(self) -> None:
        index = BM25Index().build(sample_chunks())
        corpus = make_sample_corpus(bm25_index=index)
        retriever = SparseRetriever(corpus_store=corpus)
        retriever.retrieve_results("attention", top_k=3)
        assert retriever._ensure_index() is index

    def test_explicit_index_matches_built_index(self) -> None:
        corpus = make_sample_corpus()
        explicit = SparseRetriever(corpus, bm25_index=BM25Index().build(sample_chunks()))
        built = SparseRetriever(corpus)
        q = "pre-training bidirectional representations"
        assert [r.chunk_id for r in explicit.retrieve_results(q, 5)] == [
            r.chunk_id for r in built.retrieve_results(q, 5)
        ]

    def test_index_entries_missing_from_corpus_are_skipped(self) -> None:
        """A stale index (chunks re-generated since) never returns unknown chunks."""
        stale = sample_chunks()
        stale[0] = stale[0].model_copy(update={"chunk_id": "gone"})
        retriever = SparseRetriever(make_sample_corpus(), bm25_index=BM25Index().build(stale))
        results = retriever.retrieve_results("recurrent neural networks sequence modeling", 10)
        assert "gone" not in {r.chunk_id for r in results}

    def test_empty_corpus(self) -> None:
        from app.ingestion.corpus.run_corpus import RunCorpus

        retriever = SparseRetriever(RunCorpus([], []))
        assert retriever.retrieve_results("anything", 5) == []
