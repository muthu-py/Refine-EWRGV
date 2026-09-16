"""
app/scoring/confidence.py
--------------------------
EWRGV Confidence Assessment.

Computes a confidence score in [0.0, 1.0] for a validated CandidateGap.

The confidence formula is intentionally not yet designed.  This module
holds the interface contract and a placeholder implementation.

Intended inputs to the formula (to be designed):
    - Evidence counts per type (SUPPORTING, COUNTER, PARTIAL, etc.)
    - Evidence strength scores
    - Coverage dimensions (retrieval, terminology, concept, method)
    - Corpus adequacy and uncertainty

Satisfies the ConfidenceScorer interface (app.domain.interfaces).

Status: SKELETON – formula not yet implemented.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.models.candidate_gap import CandidateGap
from app.domain.models.evidence import Evidence
from app.domain.models.validation import CoverageAssessment

logger = get_logger(__name__)


class EWRGVConfidenceScorer:
    """
    Computes the EWRGV confidence score for a validated gap.

    The formula will integrate:
        Evidence counts, evidence strength, coverage dimensions,
        and corpus adequacy into a single confidence score.

    Status: SKELETON – returns 0.0 until implemented.
    """

    def __init__(
        self,
        threshold_valid: float = 0.70,
        threshold_uncertain: float = 0.40,
    ) -> None:
        self.threshold_valid = threshold_valid
        self.threshold_uncertain = threshold_uncertain

    def score(
        self,
        gap: CandidateGap,
        evidence: list[Evidence],
        coverage: CoverageAssessment,
    ) -> float:
        """
        Return the EWRGV confidence score.
        NOT YET IMPLEMENTED – returns 0.0.
        """
        logger.warning("EWRGVConfidenceScorer.score is not yet implemented.")
        return 0.0
