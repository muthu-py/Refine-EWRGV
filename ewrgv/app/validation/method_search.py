"""
app/validation/method_search.py
---------------------------------
Method variation search.

Searches for evidence using alternative methodological framings of the gap.
For example, if the gap is about a specific technique, this search also
queries related techniques that might address the same problem.

Status: SKELETON – not implemented.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.models.candidate_gap import CandidateGap
from app.domain.models.paper import DocumentChunk

logger = get_logger(__name__)


class MethodVariationSearch:
    """
    Searches for evidence by exploring method variations and alternatives.

    Status: SKELETON – not implemented.
    """

    def __init__(self, retriever: object) -> None:
        self._retriever = retriever

    def search(self, gap: CandidateGap, queries: list[str]) -> list[DocumentChunk]:
        """Run method-variation search. NOT YET IMPLEMENTED."""
        logger.warning("MethodVariationSearch.search is not yet implemented.")
        return []
