"""
app/gap_detection/ranking.py
-----------------------------
Candidate gap ranking.

Takes a list of unranked CandidateGap objects and produces a ranked list
based on relevance, novelty, and initial confidence scores.

Status: SKELETON – not implemented.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.models.candidate_gap import CandidateGap

logger = get_logger(__name__)


class CandidateRanker:
    """
    Ranks CandidateGap objects to select the Top-K for EWRGV validation.

    Ranking criteria (to be designed):
    -----------------------------------
    * initial_confidence from the detector
    * diversity across gap types
    * number of supporting source passages
    * novelty relative to existing knowledge

    Status: SKELETON – not implemented.
    """

    def rank(
        self,
        candidates: list[CandidateGap],
        top_k: int = 5,
    ) -> list[CandidateGap]:
        """
        Rank and return the top-k candidates.
        NOT YET IMPLEMENTED – returns first top_k candidates unranked.
        """
        logger.warning("CandidateRanker.rank is not yet implemented; returning unranked top-k.")
        ranked = candidates[:top_k]
        for i, gap in enumerate(ranked):
            gap.rank = i + 1
        return ranked
