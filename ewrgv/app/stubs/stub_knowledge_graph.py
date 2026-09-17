"""
app/stubs/stub_knowledge_graph.py
-----------------------------------
Deterministic in-memory knowledge graph stub for development and testing.

.. warning::
    This is a **development stub**.  It will be replaced by a real
    knowledge graph implementation (e.g. Neo4j, or a custom graph store)
    in a future phase.

Provides a small, hand-crafted knowledge graph of ML/NLP concepts that
aligns with the mock corpus in ``stub_corpus.py``.  The graph contains:
    - ~12 entity/concept nodes
    - ~18 directed relationships
    - Entity-to-chunk mappings

Satisfies the ``KnowledgeGraphStore`` protocol (app.domain.interfaces).
"""

from __future__ import annotations


# ======================================================================
# Entity definitions
# ======================================================================

_ENTITIES: dict[str, dict] = {
    "e-transformer": {
        "entity_id": "e-transformer",
        "name": "Transformer",
        "entity_type": "architecture",
    },
    "e-self-attention": {
        "entity_id": "e-self-attention",
        "name": "Self-Attention",
        "entity_type": "mechanism",
    },
    "e-bert": {
        "entity_id": "e-bert",
        "name": "BERT",
        "entity_type": "model",
    },
    "e-gpt3": {
        "entity_id": "e-gpt3",
        "name": "GPT-3",
        "entity_type": "model",
    },
    "e-vit": {
        "entity_id": "e-vit",
        "name": "Vision Transformer",
        "entity_type": "model",
    },
    "e-rag": {
        "entity_id": "e-rag",
        "name": "Retrieval-Augmented Generation",
        "entity_type": "technique",
    },
    "e-pretraining": {
        "entity_id": "e-pretraining",
        "name": "Pre-training",
        "entity_type": "technique",
    },
    "e-fine-tuning": {
        "entity_id": "e-fine-tuning",
        "name": "Fine-tuning",
        "entity_type": "technique",
    },
    "e-nlp": {
        "entity_id": "e-nlp",
        "name": "Natural Language Processing",
        "entity_type": "domain",
    },
    "e-cv": {
        "entity_id": "e-cv",
        "name": "Computer Vision",
        "entity_type": "domain",
    },
    "e-mlm": {
        "entity_id": "e-mlm",
        "name": "Masked Language Model",
        "entity_type": "technique",
    },
    "e-dense-retrieval": {
        "entity_id": "e-dense-retrieval",
        "name": "Dense Passage Retrieval",
        "entity_type": "technique",
    },
}


# ======================================================================
# Relationships (directed edges)
# ======================================================================

_RELATIONSHIPS: list[dict] = [
    # Transformer relationships
    {"source_id": "e-transformer", "target_id": "e-self-attention",
     "relation": "uses", "target_name": "Self-Attention"},
    {"source_id": "e-transformer", "target_id": "e-nlp",
     "relation": "applied_in", "target_name": "Natural Language Processing"},
    {"source_id": "e-transformer", "target_id": "e-cv",
     "relation": "applied_in", "target_name": "Computer Vision"},
    # BERT relationships
    {"source_id": "e-bert", "target_id": "e-transformer",
     "relation": "based_on", "target_name": "Transformer"},
    {"source_id": "e-bert", "target_id": "e-pretraining",
     "relation": "uses", "target_name": "Pre-training"},
    {"source_id": "e-bert", "target_id": "e-mlm",
     "relation": "uses", "target_name": "Masked Language Model"},
    {"source_id": "e-bert", "target_id": "e-fine-tuning",
     "relation": "uses", "target_name": "Fine-tuning"},
    # GPT-3 relationships
    {"source_id": "e-gpt3", "target_id": "e-transformer",
     "relation": "based_on", "target_name": "Transformer"},
    {"source_id": "e-gpt3", "target_id": "e-pretraining",
     "relation": "uses", "target_name": "Pre-training"},
    {"source_id": "e-gpt3", "target_id": "e-nlp",
     "relation": "applied_in", "target_name": "Natural Language Processing"},
    # ViT relationships
    {"source_id": "e-vit", "target_id": "e-transformer",
     "relation": "based_on", "target_name": "Transformer"},
    {"source_id": "e-vit", "target_id": "e-cv",
     "relation": "applied_in", "target_name": "Computer Vision"},
    {"source_id": "e-vit", "target_id": "e-self-attention",
     "relation": "uses", "target_name": "Self-Attention"},
    # RAG relationships
    {"source_id": "e-rag", "target_id": "e-dense-retrieval",
     "relation": "uses", "target_name": "Dense Passage Retrieval"},
    {"source_id": "e-rag", "target_id": "e-transformer",
     "relation": "based_on", "target_name": "Transformer"},
    {"source_id": "e-rag", "target_id": "e-nlp",
     "relation": "applied_in", "target_name": "Natural Language Processing"},
    # Dense retrieval relationship
    {"source_id": "e-dense-retrieval", "target_id": "e-bert",
     "relation": "based_on", "target_name": "BERT"},
    # Pre-training → fine-tuning link
    {"source_id": "e-pretraining", "target_id": "e-fine-tuning",
     "relation": "precedes", "target_name": "Fine-tuning"},
]


# ======================================================================
# Entity-to-chunk mappings
# ======================================================================

_ENTITY_CHUNKS: dict[str, list[str]] = {
    "e-transformer": [
        "chunk-001-intro", "chunk-001-arch", "chunk-001-results",
        "chunk-004-intro",
    ],
    "e-self-attention": [
        "chunk-001-arch", "chunk-004-intro",
    ],
    "e-bert": [
        "chunk-002-intro", "chunk-002-pretrain",
    ],
    "e-gpt3": [
        "chunk-003-intro", "chunk-003-approach", "chunk-003-limits",
    ],
    "e-vit": [
        "chunk-004-intro", "chunk-004-vit",
    ],
    "e-rag": [
        "chunk-005-intro", "chunk-005-rag", "chunk-005-exp",
    ],
    "e-pretraining": [
        "chunk-002-pretrain", "chunk-003-intro",
    ],
    "e-fine-tuning": [
        "chunk-002-intro", "chunk-003-intro",
    ],
    "e-nlp": [
        "chunk-001-intro", "chunk-002-intro", "chunk-003-intro",
        "chunk-004-intro",
    ],
    "e-cv": [
        "chunk-004-intro", "chunk-004-vit",
    ],
    "e-mlm": [
        "chunk-002-pretrain",
    ],
    "e-dense-retrieval": [
        "chunk-005-rag",
    ],
}


# ======================================================================
# Name-based search index (lowercased name → entity_id)
# ======================================================================

_NAME_INDEX: dict[str, str] = {}
for _eid, _edata in _ENTITIES.items():
    # Index by full name (lowercase)
    _NAME_INDEX[_edata["name"].lower()] = _eid
    # Also index individual words for partial matching
    for _word in _edata["name"].lower().split():
        if len(_word) >= 3:  # skip very short words
            _NAME_INDEX.setdefault(_word, _eid)

# Also index some common aliases
_NAME_INDEX.update({
    "attention": "e-self-attention",
    "transformers": "e-transformer",
    "vit": "e-vit",
    "rag": "e-rag",
    "retrieval-augmented": "e-rag",
    "retrieval augmented": "e-rag",
    "gpt": "e-gpt3",
    "gpt-3": "e-gpt3",
    "gpt3": "e-gpt3",
    "dpr": "e-dense-retrieval",
    "masked language": "e-mlm",
    "computer vision": "e-cv",
    "image recognition": "e-cv",
    "language model": "e-pretraining",
    "pre-trained": "e-pretraining",
    "pretrained": "e-pretraining",
})


# Build a reverse relationship index for lookups
_RELATIONSHIPS_BY_SOURCE: dict[str, list[dict]] = {}
for _rel in _RELATIONSHIPS:
    _RELATIONSHIPS_BY_SOURCE.setdefault(_rel["source_id"], []).append(_rel)


class StubKnowledgeGraphStore:
    """
    Deterministic in-memory knowledge graph for development and testing.

    Contains 12 entities, 18 relationships, and entity-to-chunk mappings
    aligned with the mock corpus (StubCorpusStore).

    Satisfies the ``KnowledgeGraphStore`` protocol (app.domain.interfaces).

    .. warning::
        **Development stub.**  Will be replaced by a real KG implementation
        (e.g. Neo4j, RDF store) in a future phase.
    """

    def find_entities(self, query: str) -> list[dict]:
        """
        Find entities/concepts matching the query string.

        Matching strategy (deterministic):
        1. Check if any known entity name appears as a substring in the query.
        2. Check if any query token matches an indexed alias.
        3. Return all matched entities, deduplicated.
        """
        query_lower = query.lower()
        matched_ids: dict[str, None] = {}  # ordered set via dict

        # Substring match against entity names
        for entity in _ENTITIES.values():
            if entity["name"].lower() in query_lower:
                matched_ids[entity["entity_id"]] = None

        # Token match against name index
        tokens = query_lower.split()
        for token in tokens:
            token_clean = token.strip(".,;:!?()[]{}\"'")
            if token_clean in _NAME_INDEX:
                matched_ids[_NAME_INDEX[token_clean]] = None

        # Multi-word phrase match
        for phrase, eid in _NAME_INDEX.items():
            if " " in phrase and phrase in query_lower:
                matched_ids[eid] = None

        return [_ENTITIES[eid] for eid in matched_ids if eid in _ENTITIES]

    def get_related_entities(self, entity_id: str) -> list[dict]:
        """
        Get entities related to the given entity via KG edges.

        Returns all outgoing relationships from the entity.
        """
        return list(_RELATIONSHIPS_BY_SOURCE.get(entity_id, []))

    def get_entity_chunks(self, entity_id: str) -> list[str]:
        """
        Get chunk IDs associated with a given entity.
        """
        return list(_ENTITY_CHUNKS.get(entity_id, []))
