"""
app/ingestion/collectors/__init__.py
--------------------------------------
Literature collector implementations and high-level collection service.

Each collector satisfies the LiteratureCollector interface defined in
app.domain.interfaces.  The service orchestrates them.

Concrete implementations:
    SemanticScholarCollector  – Semantic Scholar API
    OpenAlexCollector         – OpenAlex API

High-level service:
    LiteratureCollectionService  – dispatches across providers + queries
"""

from app.ingestion.collectors.service import LiteratureCollectionService

__all__ = [
    "LiteratureCollectionService",
]
