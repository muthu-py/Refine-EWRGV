"""
app/ingestion/corpus/__init__.py
----------------------------------
Retrieval corpus of a research run.

Concrete implementations:
    RunCorpus        – CorpusStore over a run's PaperChunks, stored
                       embeddings and BM25 index
    RunCorpusLoader  – builds a RunCorpus from Supabase Storage + PostgreSQL
"""

from app.ingestion.corpus.run_corpus import RunCorpus, RunCorpusLoader, bm25_index_path

__all__ = ["RunCorpus", "RunCorpusLoader", "bm25_index_path"]
