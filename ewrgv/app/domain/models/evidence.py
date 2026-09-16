"""
app/domain/models/evidence.py
------------------------------
Domain model for an individual evidence item collected during EWRGV validation.

A single Paper may yield multiple Evidence items; each is independently
classified with an EvidenceType (SUPPORTING, COUNTER, PARTIAL, etc.).
This many-to-many relationship between Papers and Evidence items is
fundamental to the EWRGV scoring model.
"""

from __future__ import annotations

from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from app.domain.enums import EvidenceType


class Provenance(BaseModel):
    """
    Tracks where an evidence item came from and how it was found.

    Attributes
    ----------
    search_type:
        Which validation search produced this evidence
        (e.g. "direct", "terminology", "method_variation", "counter").
    query_used:
        The exact query string that retrieved this evidence.
    retrieval_rank:
        The rank of the parent chunk in the retrieval result list.
    rerank_score:
        The reranker score if reranking was applied.
    """

    search_type: str
    query_used: str
    retrieval_rank: Optional[int] = None
    rerank_score: Optional[float] = None


class Evidence(BaseModel):
    """
    A single piece of evidence collected during EWRGV validation for a gap.

    Attributes
    ----------
    evidence_id:
        Unique identifier for this evidence item.
    gap_id:
        The CandidateGap this evidence relates to.
    paper_id:
        The source Paper.
    passage_id:
        The DocumentChunk (passage) this evidence was extracted from.
    evidence_type:
        Classification assigned after evidence-classification step.
    passage_text:
        The actual text of the evidence passage.
    relevance:
        Relevance score (0.0–1.0) of this evidence to the gap.
    strength:
        Strength score (0.0–1.0) indicating how strongly the evidence
        supports/counters the gap.
    provenance:
        How and where this evidence was found.
    metadata:
        Free-form metadata from the classifier.
    """

    evidence_id: str = Field(default_factory=lambda: str(uuid4()))
    gap_id: str
    paper_id: str
    passage_id: str
    evidence_type: EvidenceType = EvidenceType.INSUFFICIENT
    passage_text: str = ""
    relevance: float = 0.0
    strength: float = 0.0
    provenance: Optional[Provenance] = None
    metadata: dict = Field(default_factory=dict)
