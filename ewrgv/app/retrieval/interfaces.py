"""
app/retrieval/interfaces.py
----------------------------
Retrieval-specific interface extensions and data transfer objects.

The base Retriever and Reranker protocols live in app.domain.interfaces.
This module extends them with retrieval-specific result types that carry
scores alongside the chunks (useful for fusion and reranking logic).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.domain.models.paper import DocumentChunk


@dataclass
class ScoredChunk:
    """
    A DocumentChunk paired with a retrieval score.

    Used internally within the retrieval layer.  The plain DocumentChunk
    list returned by the Retriever interface hides scores from callers
    that don't need them.
    """

    chunk: DocumentChunk
    score: float
    retriever: str = ""          # "semantic" | "bm25" | "hybrid"
    rank: int = 0


class ScoredRetriever(Protocol):
    """
    Extended retriever that returns scores alongside chunks.

    Used by the fusion module; not exposed to orchestration code.
    """

    def retrieve_scored(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        """Return scored chunks for internal fusion use."""
        ...
