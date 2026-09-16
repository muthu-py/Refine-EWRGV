"""
app/domain/models/candidate_gap.py
------------------------------------
Domain model for a Candidate Research Gap identified during gap detection.

A CandidateGap is the *output* of the gap-detection module and the *input*
to the EWRGV validation module.  This boundary is central to the research
contribution: Gap Detection ≠ Gap Validation.
"""

from __future__ import annotations

from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from app.domain.enums import GapType


class StructuredGapRepresentation(BaseModel):
    """
    A structured, decomposed representation of a gap statement.

    Produced by the gap-representation sub-component to make the gap
    amenable to systematic verification queries.
    """

    gap_statement: str
    domain: Optional[str] = None
    population: Optional[str] = None
    methodology: Optional[str] = None
    outcome: Optional[str] = None
    temporal_scope: Optional[str] = None
    geographic_scope: Optional[str] = None
    key_concepts: list[str] = Field(default_factory=list)
    synonyms: list[str] = Field(default_factory=list)


class CandidateGap(BaseModel):
    """
    A candidate research gap produced by the gap-detection step.

    Attributes
    ----------
    gap_id:
        Unique identifier for this candidate gap.
    gap_statement:
        Human-readable statement describing the gap.
    primary_gap_type:
        Primary classification from the standard gap taxonomy.
    secondary_gap_types:
        Additional taxonomy labels that also apply.
    structured_representation:
        Machine-friendly decomposition used to drive verification queries.
    source_papers:
        Paper IDs that provided evidence for this gap's identification.
    relevant_passages:
        Chunk IDs of the specific passages that motivated this gap.
    initial_confidence:
        A raw confidence score (0.0–1.0) from the detector *before* EWRGV
        validation runs.  This is deliberately distinct from the post-
        validation confidence.
    rank:
        Rank assigned by the candidate-ranking step (lower = higher priority).
    metadata:
        Free-form detector-level metadata.
    """

    gap_id: str = Field(default_factory=lambda: str(uuid4()))
    gap_statement: str
    primary_gap_type: GapType = GapType.OTHER
    secondary_gap_types: list[GapType] = Field(default_factory=list)
    structured_representation: Optional[StructuredGapRepresentation] = None
    source_papers: list[str] = Field(default_factory=list)
    relevant_passages: list[str] = Field(default_factory=list)
    initial_confidence: float = 0.0
    rank: Optional[int] = None
    metadata: dict = Field(default_factory=dict)
