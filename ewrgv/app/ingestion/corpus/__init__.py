"""
app/ingestion/corpus/__init__.py
----------------------------------
In-memory corpus construction from collected literature.

Concrete implementations:
    CollectionCorpusStore  – CorpusStore built from a CollectionResult
                             (one title + abstract chunk per unique paper)
"""

from app.ingestion.corpus.collection_corpus import CollectionCorpusStore

__all__ = ["CollectionCorpusStore"]
