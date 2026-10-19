"""
app/retrieval/interfaces.py
----------------------------
Retrieval-specific interface extensions and data transfer objects.

The base Retriever and Reranker protocols live in app.domain.interfaces.
This module extends them with retrieval-specific result types that carry
scores alongside the chunks (useful for fusion and reranking logic).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from app.domain.models.paper import Paper
from app.domain.models.paper_chunk import PaperChunk
from app.domain.models.retrieval_result import RetrievalResult


@dataclass
class ScoredChunk:
    """
    A PaperChunk paired with a retrieval score.

    Used internally within the retrieval layer.  The plain PaperChunk
    list returned by the Retriever interface hides scores from callers
    that don't need them.
    """

    chunk: PaperChunk
    score: float
    retriever: str = ""          # "dense" | "sparse" | "hybrid"
    rank: int = 0


class ScoredRetriever(Protocol):
    """
    Extended retriever that returns scores alongside chunks.

    Used by the fusion module; not exposed to orchestration code.
    """

    def retrieve_scored(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        """Return scored chunks for internal fusion use."""
        ...


def to_retrieval_result(
    scored: ScoredChunk,
    method: str,
    paper: Optional[Paper] = None,
) -> RetrievalResult:
    """Build a RetrievalResult from a scored PaperChunk (plus its paper, if known)."""
    chunk = scored.chunk
    metadata: dict = {}
    if paper is not None:
        metadata = {
            "title": paper.title,
            "publication_year": paper.publication_year,
            "doi": paper.doi,
        }
    return RetrievalResult(
        chunk_id=chunk.chunk_id,
        paper_id=chunk.paper_id,
        section=chunk.section_title,
        section_type=chunk.section_type.value,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        text=chunk.text,
        score=scored.score,
        rank=scored.rank,
        retrieval_method=method,
        metadata=metadata,
    )
