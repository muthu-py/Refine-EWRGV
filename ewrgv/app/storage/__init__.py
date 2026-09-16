"""
app/storage/__init__.py
------------------------
Storage package.

Sub-packages
------------
vector/       – VectorStore implementations (FAISS, Chroma, Qdrant adapters)
relational/   – Relational DB models and session management
repositories/ – Repository pattern implementations (PaperRepository, etc.)

All storage implementations must satisfy the interfaces defined in
app.domain.interfaces.  Domain code never imports from storage/ directly.
"""
