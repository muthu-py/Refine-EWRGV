"""
app/validation/iteration.py
-----------------------------
EWRGV iterative validation loop.

After each pass the controller checks coverage; another pass with refined
queries (broader: without the field anchor, single concepts, unused aliases)
runs while

    iteration < MAX_VALIDATION_ITERATIONS   and
    (relevant items < 5  or  retrieval coverage < 0.5  or  concept coverage < 1.0)

so a gap is never judged on searches that missed its literature.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.models.validation import CoverageAssessment

logger = get_logger(__name__)

MIN_RELEVANT = 5
MIN_RETRIEVAL_COVERAGE = 0.5


class ValidationIterationController:
    def __init__(self, max_iterations: int = 3) -> None:
        self._max_iterations = max(1, max_iterations)

    @property
    def max_iterations(self) -> int:
        return self._max_iterations

    def should_iterate(self, iteration: int, coverage: CoverageAssessment, relevant_items: int = 0) -> bool:
        if iteration >= self._max_iterations:
            return False
        if relevant_items < MIN_RELEVANT:
            return True
        if coverage.retrieval_coverage is not None and coverage.retrieval_coverage < MIN_RETRIEVAL_COVERAGE:
            return True
        return coverage.concept_coverage is not None and coverage.concept_coverage < 1.0

    def refine_queries(self, generator, gap: dict, used: set[str]):
        """Refined VerificationQueries for the next pass (see VerificationQueryGenerator.refine)."""
        return generator.refine(gap, used)
