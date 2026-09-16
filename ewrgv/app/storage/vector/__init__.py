"""
app/storage/vector/__init__.py
--------------------------------
Vector store adapter implementations.

Planned adapters:
    FAISSVectorStore   – local, in-memory/disk FAISS index
    ChromaVectorStore  – Chroma persistent store
    QdrantVectorStore  – Qdrant server or in-process

All adapters must satisfy the VectorStore protocol (app.domain.interfaces).
"""
