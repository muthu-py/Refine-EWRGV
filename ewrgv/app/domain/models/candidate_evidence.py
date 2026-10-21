"""
app/domain/models/candidate_evidence.py
-----------------------------------------
Domain models for the evidence-weighted scoring of a CandidateGap.

These describe the evidence retrieved for a *candidate* gap BEFORE EWRGV
validation.  An assessment is a prioritisation signal, not a validation
outcome: ``is_validated`` is always False, and the validation module
(app/validation/) remains solely responsible for GapClassification.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.domain.enums import CandidateEvidenceStatus, EvidenceType


class CandidateEvidenceItem(BaseModel):
    """
    One retrieved passage used as evidence for or against a candidate gap.

    Attributes
    ----------
    chunk_id, paper_id, section, section_type, page_start, page_end:
        Reference to the retrieved passage (section title, SectionType
        value and PDF page range from the PaperChunk).
    stance:
        ``EvidenceType.SUPPORTING`` (indicates the gap exists, e.g. a stated
        limitation) or ``EvidenceType.COUNTER`` (suggests it is addressed).
    relevance:
        Lexical overlap between the gap's terms and the passage, in [0, 1].
    quality:
        Section-based specificity in [0, 1] (supporting evidence only;
        ``None`` for counter evidence).
    recency:
        Publication recency in [0, 1]; ``None`` when no year is available.
    strength:
        Weighted mean of the available item features, in [0, 1].
    publication_year, title, doi, citation_count:
        Bibliographic context when available.  ``citation_count`` is
        informational only and is NOT used in any score.
    """

    chunk_id: str
    paper_id: str
    section: str = ""
    section_type: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    stance: EvidenceType
    relevance: float
    quality: Optional[float] = None
    recency: Optional[float] = None
    strength: float
    publication_year: Optional[int] = None
    title: Optional[str] = None
    doi: Optional[str] = None
    citation_count: Optional[int] = None


class CandidateEvidenceAssessment(BaseModel):
    """
    Evidence-weighted score for one CandidateGap (pre-validation).

    Attributes
    ----------
    score:
        Final evidence score in [0, 1]:
        ``support_score × (1 − contradiction_penalty × contradiction_ratio)``;
        0.0 when there is no supporting evidence.
    status:
        CandidateEvidenceStatus summarising the evidence situation.
    support_score:
        Weighted mean of the available candidate-level features.
    contradiction_ratio:
        Counter-evidence share of total evidence strength, in [0, 1].
    features:
        Candidate-level feature values (``relevance``, ``quality``,
        ``recency``, ``independence``); ``None`` = unavailable, excluded.
    weights:
        Effective (renormalised) weights of the features that were used.
    supporting, contradicting:
        Evidence references.
    missing_chunk_ids:
        Cited passages that were not among the available chunks.
    explanation:
        Human-readable account of how the score was obtained.
    is_validated:
        Always False — this is not EWRGV validation.
    """

    score: float
    status: CandidateEvidenceStatus
    support_score: float
    contradiction_ratio: float
    features: dict[str, Optional[float]] = Field(default_factory=dict)
    weights: dict[str, float] = Field(default_factory=dict)
    supporting: list[CandidateEvidenceItem] = Field(default_factory=list)
    contradicting: list[CandidateEvidenceItem] = Field(default_factory=list)
    missing_chunk_ids: list[str] = Field(default_factory=list)
    explanation: str = ""
    is_validated: Literal[False] = False
