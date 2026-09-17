"""
app/retrieval/__init__.py
--------------------------
Hybrid retrieval package.

Responsibilities
----------------
* Dense (semantic) retrieval   – app/retrieval/semantic.py
* Sparse (BM25) retrieval      – app/retrieval/bm25.py
* Knowledge Graph retrieval    – app/retrieval/knowledge_graph.py
* Reciprocal Rank Fusion       – app/retrieval/fusion.py
* Reranking                    – app/retrieval/reranking.py

All retrievers satisfy the Retriever interface (app.domain.interfaces).
The HybridRetriever composes DenseRetriever + SparseRetriever +
KnowledgeGraphRetriever via Reciprocal Rank Fusion (RRF).
"""
