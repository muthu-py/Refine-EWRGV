"""
app/providers/__init__.py
--------------------------
External provider adapters package.

Each sub-package contains concrete implementations of domain interfaces
for a specific external service.

Sub-packages
------------
llm/         – LLMProvider implementations
embeddings/  – EmbeddingProvider implementations
search/      – LiteratureCollector implementations

Application and domain code must NEVER import from this package directly.
All access is through the interfaces in app.domain.interfaces.
"""
