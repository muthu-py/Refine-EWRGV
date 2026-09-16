"""
app/validation/direct_search.py
---------------------------------
Direct evidence search: retrieves passages that directly address the gap claim.

This is the first EWRGV validation search type.  It uses the gap statement
and its key concepts verbatim to find the most obviously relevant evidence.

Status: SKELETON – not implemented.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.models.candidate_gap import CandidateGap
from app.domain.models.paper import DocumentChunk

logger = get_logger(__name__)


class DirectEvidenceSearch:
    """
    Executes direct retrieval for a CandidateGap.

    Status: SKELETON – not implemented.
    """

    def __init__(self, retriever: object) -> None:
        self._retriever = retriever

    def search(self, gap: CandidateGap, queries: list[str]) -> list[DocumentChunk]:
        """Run direct evidence search. NOT YET IMPLEMENTED."""
        logger.warning("DirectEvidenceSearch.search is not yet implemented.")
        return []
