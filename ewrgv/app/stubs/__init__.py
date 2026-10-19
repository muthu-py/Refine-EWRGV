"""
app/stubs/__init__.py
----------------------
Temporary stub implementations for development and testing.

.. warning::
    All modules in this package are **development stubs**.  They provide
    deterministic, in-memory implementations of domain interfaces so that
    retrieval logic can be developed and tested without external services.

    These stubs will be replaced by production implementations when:
    - Corpus Storage → real database-backed store (Phase 3+)
    - Text Representation → real embedding API (OpenAI, Sentence-Transformers)
    - Knowledge Graph → real KG database (Neo4j, or similar)
"""
