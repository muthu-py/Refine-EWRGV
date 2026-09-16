"""
app/validation/counter_search.py
----------------------------------
Mandatory counter-evidence search.

This is a *required* step in EWRGV.  The system must actively look for
evidence that challenges or refutes the identified gap.  This prevents
false positives where a gap appears to exist simply because the initial
retrieval didn't surface conflicting work.

Status: SKELETON – not implemented.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.models.candidate_gap import CandidateGap
from app.domain.models.paper import DocumentChunk

logger = get_logger(__name__)


class CounterEvidenceSearch:
    """
    Actively searches for counter-evidence against a CandidateGap.

    This is a *mandatory* step — skipping it would undermine the
    validity of the EWRGV process.

    Status: SKELETON – not implemented.
    """

    def __init__(self, retriever: object) -> None:
        self._retriever = retriever

    def search(self, gap: CandidateGap, queries: list[str]) -> list[DocumentChunk]:
        """
        Search specifically for evidence that contradicts the gap claim.
        NOT YET IMPLEMENTED.
        """
        logger.warning("CounterEvidenceSearch.search is not yet implemented.")
        return []
