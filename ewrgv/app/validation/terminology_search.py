"""
app/validation/terminology_search.py
--------------------------------------
Terminology / synonym expansion search.

Uses synonyms, acronyms, and alternative terms for the gap's key concepts
to find evidence that would be missed by verbatim search.

Status: SKELETON – not implemented.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.models.candidate_gap import CandidateGap
from app.domain.models.paper import DocumentChunk

logger = get_logger(__name__)


class TerminologySearch:
    """
    Searches for evidence using terminology expansion.

    Status: SKELETON – not implemented.
    """

    def __init__(self, retriever: object) -> None:
        self._retriever = retriever

    def search(self, gap: CandidateGap, queries: list[str]) -> list[DocumentChunk]:
        """Run terminology-expanded search. NOT YET IMPLEMENTED."""
        logger.warning("TerminologySearch.search is not yet implemented.")
        return []
