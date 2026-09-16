"""
app/retrieval/__init__.py
--------------------------
Hybrid retrieval package.

Responsibilities
----------------
* Semantic (dense) retrieval   – app/retrieval/semantic.py
* BM25 (sparse) retrieval      – app/retrieval/bm25.py
* Reciprocal Rank Fusion       – app/retrieval/fusion.py
* Reranking                    – app/retrieval/reranking.py

All retrievers satisfy the Retriever interface (app.domain.interfaces).
The HybridRetriever composes SemanticRetriever + BM25Retriever via fusion.
"""
