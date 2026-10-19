"""
tests/unit/test_kg_retrieval.py
---------------------------------
Unit tests for KnowledgeGraphRetriever.

Verifies:
    - Concept/entity matching works
    - Related chunks are returned
    - Respects top_k
    - Correct retrieval_method
    - Metadata includes matched entities and relations
    - Deterministic behavior
"""

from __future__ import annotations

from app.retrieval.knowledge_graph import KnowledgeGraphRetriever
from app.stubs.stub_corpus import StubCorpusStore
from app.stubs.stub_knowledge_graph import StubKnowledgeGraphStore


class TestKnowledgeGraphRetriever:
    """Tests for knowledge graph retrieval."""

    def setup_method(self) -> None:
        self.corpus = StubCorpusStore()
        self.kg = StubKnowledgeGraphStore()
        self.retriever = KnowledgeGraphRetriever(
            kg_store=self.kg,
            corpus_store=self.corpus,
        )

    # ------------------------------------------------------------------ #
    # Basic functionality
    # ------------------------------------------------------------------ #

    def test_returns_results(self) -> None:
        """A query with matching concepts returns results."""
        results = self.retriever.retrieve_results("Transformer attention", top_k=5)
        assert len(results) > 0

    def test_returns_retrieval_result_objects(self) -> None:
        """Results are RetrievalResult objects with required fields."""
        results = self.retriever.retrieve_results("BERT pre-training", top_k=3)
        for r in results:
            assert r.chunk_id
            assert r.paper_id
            assert r.text
            assert r.retrieval_method == "knowledge_graph"
            assert r.rank > 0
            assert isinstance(r.score, float)

    # ------------------------------------------------------------------ #
    # Concept matching
    # ------------------------------------------------------------------ #

    def test_transformer_query_returns_transformer_chunks(self) -> None:
        """Querying 'Transformer' returns chunks from the Transformer paper."""
        results = self.retriever.retrieve_results("Transformer", top_k=10)
        chunk_ids = [r.chunk_id for r in results]
        # Direct Transformer chunks should appear
        assert any("001" in cid for cid in chunk_ids)

    def test_bert_query_returns_bert_chunks(self) -> None:
        """Querying 'BERT' returns chunks from the BERT paper."""
        results = self.retriever.retrieve_results("BERT", top_k=10)
        chunk_ids = [r.chunk_id for r in results]
        assert any("002" in cid for cid in chunk_ids)

    def test_rag_query_returns_rag_chunks(self) -> None:
        """Querying 'RAG' returns chunks from the RAG paper."""
        results = self.retriever.retrieve_results("retrieval augmented generation", top_k=10)
        chunk_ids = [r.chunk_id for r in results]
        assert any("005" in cid for cid in chunk_ids)

    # ------------------------------------------------------------------ #
    # top_k
    # ------------------------------------------------------------------ #

    def test_respects_top_k(self) -> None:
        """Never returns more than top_k results."""
        results = self.retriever.retrieve_results("Transformer", top_k=3)
        assert len(results) <= 3

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
    # Metadata
    # ------------------------------------------------------------------ #

    def test_metadata_includes_entities(self) -> None:
        """Results include matched_entities in metadata."""
        results = self.retriever.retrieve_results("Transformer", top_k=5)
        # At least one result should have entity info
        has_entity_info = any(
            r.metadata.get("matched_entities") or r.metadata.get("relations")
            for r in results
        )
        assert has_entity_info

    # ------------------------------------------------------------------ #
    # Scores
    # ------------------------------------------------------------------ #

    def test_scores_in_valid_range(self) -> None:
        """Scores are in [0, 1] after normalisation."""
        results = self.retriever.retrieve_results("Transformer BERT", top_k=10)
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_scores_are_descending(self) -> None:
        """Results are sorted by score descending."""
        results = self.retriever.retrieve_results("self-attention", top_k=10)
        scores = [r.score for r in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    # ------------------------------------------------------------------ #
    # Edge cases
    # ------------------------------------------------------------------ #

    def test_no_matching_concepts(self) -> None:
        """Query with no matching KG concepts returns empty results."""
        results = self.retriever.retrieve_results("quantum computing hardware", top_k=5)
        assert results == []

    def test_ranks_are_sequential(self) -> None:
        """Ranks are sequential starting from 1."""
        results = self.retriever.retrieve_results("BERT", top_k=5)
        for i, r in enumerate(results):
            assert r.rank == i + 1

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
