"""
tests/unit/test_stub_knowledge_graph.py
-----------------------------------------
Unit tests for StubKnowledgeGraphStore.

Verifies:
    - Entity lookup works
    - Relationship traversal works
    - Entity-to-chunk mapping works
    - Unknown entities return empty results
"""

from __future__ import annotations

from app.stubs.stub_knowledge_graph import StubKnowledgeGraphStore


class TestStubKnowledgeGraphStore:
    """Tests for the deterministic mock knowledge graph."""

    def setup_method(self) -> None:
        self.kg = StubKnowledgeGraphStore()

    # ------------------------------------------------------------------ #
    # Entity lookup
    # ------------------------------------------------------------------ #

    def test_find_entities_by_name(self) -> None:
        """Finds entities when their full name appears in the query."""
        entities = self.kg.find_entities("Transformer architecture")
        entity_names = [e["name"] for e in entities]
        assert "Transformer" in entity_names

    def test_find_entities_bert(self) -> None:
        """Finds BERT entity."""
        entities = self.kg.find_entities("How does BERT work?")
        entity_names = [e["name"] for e in entities]
        assert "BERT" in entity_names

    def test_find_entities_multiple(self) -> None:
        """Query mentioning multiple concepts returns multiple entities."""
        entities = self.kg.find_entities("Transformer and BERT models")
        entity_names = [e["name"] for e in entities]
        assert "Transformer" in entity_names
        assert "BERT" in entity_names

    def test_find_entities_case_insensitive(self) -> None:
        """Entity matching is case-insensitive."""
        e1 = self.kg.find_entities("transformer")
        e2 = self.kg.find_entities("TRANSFORMER")
        ids1 = {e["entity_id"] for e in e1}
        ids2 = {e["entity_id"] for e in e2}
        assert ids1 == ids2

    def test_find_entities_no_match(self) -> None:
        """Query with no matching entities returns empty list."""
        entities = self.kg.find_entities("quantum computing hardware")
        assert entities == []

    def test_find_entities_returns_valid_dicts(self) -> None:
        """Returned entities have required keys."""
        entities = self.kg.find_entities("Transformer")
        for e in entities:
            assert "entity_id" in e
            assert "name" in e
            assert "entity_type" in e

    # ------------------------------------------------------------------ #
    # Relationship traversal
    # ------------------------------------------------------------------ #

    def test_get_related_entities_transformer(self) -> None:
        """Transformer has outgoing relationships."""
        related = self.kg.get_related_entities("e-transformer")
        assert len(related) > 0
        # Transformer uses Self-Attention
        relations = {r["relation"] for r in related}
        assert "uses" in relations

    def test_get_related_entities_bert(self) -> None:
        """BERT is based on Transformer."""
        related = self.kg.get_related_entities("e-bert")
        target_ids = [r["target_id"] for r in related]
        assert "e-transformer" in target_ids

    def test_get_related_entities_unknown(self) -> None:
        """Unknown entity returns empty list."""
        assert self.kg.get_related_entities("nonexistent") == []

    def test_relationship_has_required_keys(self) -> None:
        """Relationships have the required keys."""
        related = self.kg.get_related_entities("e-transformer")
        for r in related:
            assert "source_id" in r
            assert "target_id" in r
            assert "relation" in r

    # ------------------------------------------------------------------ #
    # Entity-to-chunk mapping
    # ------------------------------------------------------------------ #

    def test_get_entity_chunks_transformer(self) -> None:
        """Transformer entity has linked chunks."""
        chunks = self.kg.get_entity_chunks("e-transformer")
        assert len(chunks) > 0
        # All values should be chunk ID strings
        for cid in chunks:
            assert isinstance(cid, str)
            assert cid.startswith("chunk-")

    def test_get_entity_chunks_bert(self) -> None:
        """BERT entity links to paper-002 chunks."""
        chunks = self.kg.get_entity_chunks("e-bert")
        assert any("002" in cid for cid in chunks)

    def test_get_entity_chunks_unknown(self) -> None:
        """Unknown entity returns empty list."""
        assert self.kg.get_entity_chunks("nonexistent") == []

    # ------------------------------------------------------------------ #
    # Determinism
    # ------------------------------------------------------------------ #

    def test_deterministic_entity_lookup(self) -> None:
        """Same query always returns same entities."""
        e1 = self.kg.find_entities("Transformer attention mechanism")
        e2 = self.kg.find_entities("Transformer attention mechanism")
        assert [e["entity_id"] for e in e1] == [e["entity_id"] for e in e2]

    def test_deterministic_related(self) -> None:
        """Same entity always returns same relationships."""
        r1 = self.kg.get_related_entities("e-bert")
        r2 = self.kg.get_related_entities("e-bert")
        assert r1 == r2
