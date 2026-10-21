"""
app/retrieval/__init__.py
--------------------------
Hybrid retrieval package.

Responsibilities
----------------
* Dense (semantic) retrieval   – app/retrieval/semantic.py
* Sparse (BM25) retrieval      – app/retrieval/bm25.py
* Persisted BM25 index         – app/retrieval/bm25_index.py
* Retrieval service            – app/retrieval/service.py
* Reciprocal Rank Fusion       – app/retrieval/fusion.py
* Reranking                    – app/retrieval/reranking.py

All retrievers satisfy the Retriever interface (app.domain.interfaces).
The HybridRetriever composes DenseRetriever + SparseRetriever via
Reciprocal Rank Fusion (RRF).  All retrievers work on PaperChunks from
a run's corpus (app.ingestion.corpus.RunCorpus).
"""
