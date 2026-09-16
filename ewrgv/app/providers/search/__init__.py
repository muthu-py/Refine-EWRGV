"""
app/providers/search/__init__.py
----------------------------------
Literature search provider implementations (LiteratureCollector).

Concrete providers:
    SemanticScholarCollector  – Semantic Scholar Academic Graph API
    OpenAlexCollector         – OpenAlex API
"""

from app.providers.search.semantic_scholar import SemanticScholarCollector
from app.providers.search.openalex import OpenAlexCollector

__all__ = [
    "SemanticScholarCollector",
    "OpenAlexCollector",
]
