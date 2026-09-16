"""
app/explanation/generator.py
------------------------------
Explanation generator: produces a natural-language narrative explaining the
EWRGV validation result for a CandidateGap.

Satisfies the ExplanationGenerator interface (app.domain.interfaces).

Status: SKELETON – not implemented.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.models.candidate_gap import CandidateGap
from app.domain.models.validation import ValidationResult

logger = get_logger(__name__)


class LLMExplanationGenerator:
    """
    Generates a natural-language explanation using an LLM.

    The explanation includes:
        - Gap statement
        - Evidence summary (supporting vs. counter)
        - Coverage assessment highlights
        - Confidence score rationale
        - Final classification with justification

    Status: SKELETON – not implemented.
    """

    def __init__(self, llm_provider: object) -> None:
        self._llm = llm_provider

    def generate(self, gap: CandidateGap, result: ValidationResult) -> str:
        """Generate an explanation narrative. NOT YET IMPLEMENTED."""
        logger.warning("LLMExplanationGenerator.generate is not yet implemented.")
        return (
            f"[PLACEHOLDER] Gap '{gap.gap_statement[:60]}...' "
            f"classified as {result.classification} "
            f"(confidence: {result.confidence})."
        )
