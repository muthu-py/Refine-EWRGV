"""
app/domain/models/collection.py
--------------------------------
Domain models for literature collection results (Phase 2.2).

These models represent the output of the automatic literature collection
stage and carry provenance information so that:

1. Downstream deduplication (Phase 2.3) can merge duplicate discoveries
   without losing their original sources.
2. The API can surface which provider/query produced each paper.

Design principles
-----------------
* ``PaperProvenance`` is kept separate from ``Paper`` intentionally.
  The same canonical paper may have been discovered by multiple queries
  or multiple providers.  Provenance is a property of the *discovery
  event*, not the paper identity.
* ``CollectedPaper`` is the atomic unit returned by a collector — it
  pairs the canonical Paper with a single discovery provenance record.
* ``CollectionResult`` is the aggregate output of the entire collection
  stage for one ResearchQuery.
"""

from __future__ import annotations

from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from app.domain.models.paper import Paper


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #


class PaperProvenance(BaseModel):
    """
    Records *how* a paper was discovered during literature collection.

    Attributes
    ----------
    source:
        Provider that returned this paper (e.g. ``"semantic_scholar"``,
        ``"openalex"``).
    provider_id:
        The paper's native identifier inside that provider
        (e.g. Semantic Scholar ``paperId``, OpenAlex ``work_id``).
    source_query:
        The exact search query string that produced this result.
    """

    source: str
    provider_id: str
    source_query: str


# --------------------------------------------------------------------------- #
# Collected paper (paper + one provenance record)
# --------------------------------------------------------------------------- #


class CollectedPaper(BaseModel):
    """
    The atomic unit returned by a literature collector.

    Pairs a canonical ``Paper`` domain object with the provenance of its
    discovery.  If the same paper was found by multiple queries or
    providers, there will be multiple ``CollectedPaper`` instances with
    the same ``paper.doi`` (or ``paper.title``), each carrying its own
    ``provenance``.  Phase 2.3 merges these.

    Attributes
    ----------
    paper:
        The normalised canonical Paper.
    provenance:
        How and where this paper was found.
    """

    paper: Paper
    provenance: PaperProvenance


# --------------------------------------------------------------------------- #
# Collection result
# --------------------------------------------------------------------------- #


class CollectionResult(BaseModel):
    """
    Aggregate result from the ``LiteratureCollectionService`` for one
    ``ResearchQuery``.

    Attributes
    ----------
    query_id:
        The ``ResearchQuery.query_id`` this result belongs to.
    queries_searched:
        The exact expanded-query strings that were searched.
    providers_used:
        Providers that were invoked (may be a subset of configured
        providers if some were disabled or failed).
    papers:
        All ``CollectedPaper`` objects across all providers and queries,
        including duplicates.  Deduplication is Phase 2.3.
    provider_errors:
        List of recorded provider failures.  Each item contains at
        minimum ``{"provider": str, "query": str, "error": str}``.
    status:
        Overall status of the collection.
        * ``"success"``  – at least one paper collected, no provider errors
        * ``"partial"``  – papers collected, but some provider errors occurred
        * ``"failed"``   – all providers failed; no papers collected
    total_collected:
        Convenience count: ``len(papers)``.
    """

    query_id: str
    queries_searched: list[str] = Field(default_factory=list)
    providers_used: list[str] = Field(default_factory=list)
    papers: list[CollectedPaper] = Field(default_factory=list)
    provider_errors: list[dict] = Field(default_factory=list)
    status: str = "success"

    @property
    def total_collected(self) -> int:
        """Number of collected papers (including duplicates)."""
        return len(self.papers)
