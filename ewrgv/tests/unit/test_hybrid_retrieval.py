"""
tests/unit/test_hybrid_retrieval.py
-------------------------------------
Unit tests for HybridRetriever and RRF fusion.

Verifies:
    - Combines all three retrieval methods
    - RRF is applied correctly
    - Removes duplicate chunks
    - Preserves provenance
    - Respects final top_k
    - Deterministic ordering
    - Works with 2 methods (without KG)
"""

from __future__ import annotations

from app.domain.models.retrieval_result import RetrievalResult
from app.retrieval.bm25 import SparseRetriever
from app.retrieval.fusion import (
    DEFAULT_RRF_K,
    HybridRetriever,
    reciprocal_rank_fusion_results,
)
from app.retrieval.knowledge_graph import KnowledgeGraphRetriever
from app.retrieval.semantic import DenseRetriever
from app.stubs.stub_corpus import StubCorpusStore
from app.stubs.stub_knowledge_graph import StubKnowledgeGraphStore
from app.stubs.stub_text_representation import StubEmbeddingProvider


def _build_full_hybrid() -> HybridRetriever:
    """Build a HybridRetriever with all three methods."""
    corpus = StubCorpusStore()
    embedder = StubEmbeddingProvider()
    kg = StubKnowledgeGraphStore()

    dense = DenseRetriever(embedding_provider=embedder, corpus_store=corpus)
    sparse = SparseRetriever(corpus_store=corpus)
    kg_retriever = KnowledgeGraphRetriever(kg_store=kg, corpus_store=corpus)

    return HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        kg_retriever=kg_retriever,
        top_k=10,
        rrf_k=DEFAULT_RRF_K,
    )


class TestReciprocralRankFusion:
    """Tests for the RRF fusion function itself."""

    def test_single_list(self) -> None:
        """RRF with a single list preserves the ranking."""
        results = [
            RetrievalResult(chunk_id="c1", paper_id="p1", text="t1", score=0.9, rank=1, retrieval_method="dense"),
            RetrievalResult(chunk_id="c2", paper_id="p1", text="t2", score=0.8, rank=2, retrieval_method="dense"),
            RetrievalResult(chunk_id="c3", paper_id="p1", text="t3", score=0.7, rank=3, retrieval_method="dense"),
        ]
        fused = reciprocal_rank_fusion_results([results], top_k=10)
        assert len(fused) == 3
        assert fused[0].chunk_id == "c1"
        assert fused[1].chunk_id == "c2"
        assert fused[2].chunk_id == "c3"

    def test_rrf_formula(self) -> None:
        """RRF scores follow the formula: score = Σ 1/(k + rank)."""
        k = 60
        list1 = [
            RetrievalResult(chunk_id="c1", paper_id="p1", text="t1", score=1.0, rank=1, retrieval_method="dense"),
        ]
        list2 = [
            RetrievalResult(chunk_id="c1", paper_id="p1", text="t1", score=1.0, rank=1, retrieval_method="sparse"),
        ]
        fused = reciprocal_rank_fusion_results([list1, list2], top_k=10, k=k)
        expected = 1.0 / (k + 1) + 1.0 / (k + 1)  # rank 1 in both lists
        assert len(fused) == 1
        assert abs(fused[0].score - expected) < 1e-10

    def test_deduplication(self) -> None:
        """Duplicate chunk_ids across lists are merged, not duplicated."""
        list1 = [
            RetrievalResult(chunk_id="c1", paper_id="p1", text="t1", score=0.9, rank=1, retrieval_method="dense"),
            RetrievalResult(chunk_id="c2", paper_id="p1", text="t2", score=0.8, rank=2, retrieval_method="dense"),
        ]
        list2 = [
            RetrievalResult(chunk_id="c1", paper_id="p1", text="t1", score=0.7, rank=1, retrieval_method="sparse"),
            RetrievalResult(chunk_id="c3", paper_id="p1", text="t3", score=0.6, rank=2, retrieval_method="sparse"),
        ]
        fused = reciprocal_rank_fusion_results([list1, list2], top_k=10)
        chunk_ids = [r.chunk_id for r in fused]
        assert len(chunk_ids) == len(set(chunk_ids)), "No duplicates in fused results"
        assert len(fused) == 3  # c1, c2, c3

    def test_top_k_respected(self) -> None:
        """RRF respects the top_k parameter."""
        results = [
            RetrievalResult(chunk_id=f"c{i}", paper_id="p1", text=f"t{i}", score=1.0, rank=i, retrieval_method="dense")
            for i in range(1, 11)
        ]
        fused = reciprocal_rank_fusion_results([results], top_k=3)
        assert len(fused) == 3

    def test_provenance_preserved(self) -> None:
        """Fused results include contributing_methods in metadata."""
        list1 = [
            RetrievalResult(chunk_id="c1", paper_id="p1", text="t1", score=0.9, rank=1, retrieval_method="dense"),
        ]
        list2 = [
            RetrievalResult(chunk_id="c1", paper_id="p1", text="t1", score=0.7, rank=2, retrieval_method="sparse"),
        ]
        fused = reciprocal_rank_fusion_results([list1, list2], top_k=10)
        assert len(fused) == 1
        methods = fused[0].metadata.get("contributing_methods", [])
        method_names = [m["method"] for m in methods]
        assert "dense" in method_names
        assert "sparse" in method_names

    def test_retrieval_method_is_hybrid(self) -> None:
        """All fused results have retrieval_method='hybrid'."""
        list1 = [
            RetrievalResult(chunk_id="c1", paper_id="p1", text="t1", score=0.9, rank=1, retrieval_method="dense"),
        ]
        fused = reciprocal_rank_fusion_results([list1], top_k=10)
        for r in fused:
            assert r.retrieval_method == "hybrid"

    def test_deterministic_tie_breaking(self) -> None:
        """When scores are tied, chunk_id is used as tiebreaker."""
        list1 = [
            RetrievalResult(chunk_id="c2", paper_id="p1", text="t2", score=0.9, rank=1, retrieval_method="dense"),
            RetrievalResult(chunk_id="c1", paper_id="p1", text="t1", score=0.8, rank=2, retrieval_method="dense"),
        ]
        list2 = [
            RetrievalResult(chunk_id="c1", paper_id="p1", text="t1", score=0.9, rank=1, retrieval_method="sparse"),
            RetrievalResult(chunk_id="c2", paper_id="p1", text="t2", score=0.8, rank=2, retrieval_method="sparse"),
        ]
        # Both c1 and c2 will have the same RRF score (1/(k+1) + 1/(k+2))
        fused = reciprocal_rank_fusion_results([list1, list2], top_k=10)
        # With equal scores, should sort by chunk_id alphabetically
        assert fused[0].chunk_id == "c1"
        assert fused[1].chunk_id == "c2"


class TestHybridRetriever:
    """Tests for the HybridRetriever class."""

    def setup_method(self) -> None:
        self.hybrid = _build_full_hybrid()

    # ------------------------------------------------------------------ #
    # Basic functionality
    # ------------------------------------------------------------------ #

    def test_returns_results(self) -> None:
        """A query returns non-empty fused results."""
        results = self.hybrid.retrieve("Transformer attention", top_k=5)
        assert len(results) > 0

    def test_all_results_are_hybrid(self) -> None:
        """All results have retrieval_method='hybrid'."""
        results = self.hybrid.retrieve("language model", top_k=5)
        for r in results:
            assert r.retrieval_method == "hybrid"

    # ------------------------------------------------------------------ #
    # top_k
    # ------------------------------------------------------------------ #

    def test_respects_top_k(self) -> None:
        """Never returns more than top_k results."""
        results = self.hybrid.retrieve("Transformer", top_k=3)
        assert len(results) <= 3

    # ------------------------------------------------------------------ #
    # Deduplication
    # ------------------------------------------------------------------ #

    def test_no_duplicate_chunks(self) -> None:
        """No duplicate chunk_ids in fused results."""
        results = self.hybrid.retrieve("Transformer model", top_k=20)
        chunk_ids = [r.chunk_id for r in results]
        assert len(chunk_ids) == len(set(chunk_ids))

    # ------------------------------------------------------------------ #
    # Determinism
    # ------------------------------------------------------------------ #

    def test_deterministic_ordering(self) -> None:
        """Same query produces same ordering every time."""
        r1 = self.hybrid.retrieve("BERT pre-training", top_k=10)
        r2 = self.hybrid.retrieve("BERT pre-training", top_k=10)
        assert [r.chunk_id for r in r1] == [r.chunk_id for r in r2]
        assert [r.score for r in r1] == [r.score for r in r2]

    # ------------------------------------------------------------------ #
    # Provenance
    # ------------------------------------------------------------------ #

    def test_provenance_metadata(self) -> None:
        """Results include contributing_methods metadata."""
        results = self.hybrid.retrieve("Transformer", top_k=5)
        for r in results:
            assert "contributing_methods" in r.metadata
            assert len(r.metadata["contributing_methods"]) > 0

    # ------------------------------------------------------------------ #
    # Ranks
    # ------------------------------------------------------------------ #

    def test_ranks_are_sequential(self) -> None:
        """Ranks are sequential starting from 1."""
        results = self.hybrid.retrieve("attention mechanism", top_k=5)
        for i, r in enumerate(results):
            assert r.rank == i + 1

    # ------------------------------------------------------------------ #
    # Without KG
    # ------------------------------------------------------------------ #

    def test_works_without_kg(self) -> None:
        """HybridRetriever works with just dense + sparse (no KG)."""
        corpus = StubCorpusStore()
        embedder = StubEmbeddingProvider()
        dense = DenseRetriever(embedding_provider=embedder, corpus_store=corpus)
        sparse = SparseRetriever(corpus_store=corpus)

        hybrid = HybridRetriever(
            dense_retriever=dense,
            sparse_retriever=sparse,
            kg_retriever=None,
            top_k=5,
        )
        results = hybrid.retrieve("Transformer model")
        assert len(results) > 0
        for r in results:
            assert r.retrieval_method == "hybrid"

    # ------------------------------------------------------------------ #
    # RRF constant
    # ------------------------------------------------------------------ #

    def test_custom_rrf_k(self) -> None:
        """Custom RRF k is passed through to metadata."""
        corpus = StubCorpusStore()
        embedder = StubEmbeddingProvider()
        dense = DenseRetriever(embedding_provider=embedder, corpus_store=corpus)
        sparse = SparseRetriever(corpus_store=corpus)

        hybrid = HybridRetriever(
            dense_retriever=dense,
            sparse_retriever=sparse,
            top_k=5,
            rrf_k=30,
        )
        results = hybrid.retrieve("Transformer")
        assert len(results) > 0
        assert results[0].metadata.get("rrf_k") == 30
