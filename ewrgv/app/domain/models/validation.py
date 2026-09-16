"""
app/domain/models/validation.py
--------------------------------
Domain models for EWRGV validation outputs.

These are the *result* types produced after the full EWRGV validation
pipeline runs on a CandidateGap.  They carry all the information needed
to explain and justify the final gap classification.

Intentionally separated from candidate_gap.py because:
    CandidateGap  →  (EWRGV Validation)  →  ValidationResult
is the central contribution of this research system.
"""

from __future__ import annotations

from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from app.domain.enums import GapClassification
from app.domain.models.evidence import Evidence


class CoverageAssessment(BaseModel):
    """
    Multi-dimensional coverage assessment produced by the coverage-analysis step.

    Each dimension is scored 0.0–1.0. None means the dimension has not yet
    been computed (assessment is incremental during validation iterations).

    Attributes
    ----------
    retrieval_coverage:
        How well the retrieved evidence covers the gap's topic space.
    concept_coverage:
        Coverage of key concepts identified in the structured gap representation.
    terminology_coverage:
        Coverage after synonym / alternative-terminology search.
    method_coverage:
        Coverage after method-variation search.
    corpus_adequacy:
        Estimated adequacy of the underlying corpus for this gap topic.
    corpus_uncertainty:
        Uncertainty introduced by potential corpus gaps or biases.
    """

    retrieval_coverage: Optional[float] = None
    concept_coverage: Optional[float] = None
    terminology_coverage: Optional[float] = None
    method_coverage: Optional[float] = None
    corpus_adequacy: Optional[float] = None
    corpus_uncertainty: Optional[float] = None
    notes: list[str] = Field(default_factory=list)


class EvidenceChainEntry(BaseModel):
    """
    A single step in the human-readable evidence chain explanation.
    """

    step: int
    label: str           # e.g. "Direct Search", "Counter Evidence"
    summary: str
    evidence_ids: list[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """
    The final output of the EWRGV validation process for one CandidateGap.

    Attributes
    ----------
    result_id:
        Unique identifier for this validation result.
    gap_id:
        The CandidateGap this result corresponds to.
    evidence:
        All evidence items collected during validation.
    coverage:
        Multi-dimensional coverage assessment.
    confidence:
        Final confidence score (0.0–1.0) from the confidence-assessment step.
    classification:
        VALID_GAP | UNCERTAIN_GAP | UNSUPPORTED_GAP
    explanation:
        Narrative explanation produced by the explanation generator.
    evidence_chain:
        Step-by-step evidence chain supporting the classification.
    validation_iterations:
        Number of EWRGV iteration loops performed.
    metadata:
        Metadata from the validator (e.g. models used, timestamps).
    """

    result_id: str = Field(default_factory=lambda: str(uuid4()))
    gap_id: str
    evidence: list[Evidence] = Field(default_factory=list)
    coverage: Optional[CoverageAssessment] = None
    confidence: Optional[float] = None
    classification: Optional[GapClassification] = None
    explanation: str = ""
    evidence_chain: list[EvidenceChainEntry] = Field(default_factory=list)
    validation_iterations: int = 0
    metadata: dict = Field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience helpers (pure domain logic — no I/O, no dependencies)
    # ------------------------------------------------------------------

    @property
    def supporting_count(self) -> int:
        from app.domain.enums import EvidenceType
        return sum(1 for e in self.evidence if e.evidence_type == EvidenceType.SUPPORTING)

    @property
    def counter_count(self) -> int:
        from app.domain.enums import EvidenceType
        return sum(1 for e in self.evidence if e.evidence_type == EvidenceType.COUNTER)

    @property
    def is_classified(self) -> bool:
        return self.classification is not None
