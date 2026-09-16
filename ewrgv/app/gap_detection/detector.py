"""
app/gap_detection/detector.py
-------------------------------
Gap detector: identifies candidate research gaps from retrieved evidence.

The detector satisfies the GapDetector interface (app.domain.interfaces).
It receives retrieved DocumentChunks and a ResearchQuery, and produces
a list of CandidateGap domain objects.

The detected gaps are *unvalidated*.  All downstream confidence and
classification is the responsibility of the EWRGV validation module.

Status: SKELETON – not implemented.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.models.candidate_gap import CandidateGap
from app.domain.models.paper import DocumentChunk
from app.domain.models.query import ResearchQuery

logger = get_logger(__name__)


class GapDetector:
    """
    Identifies candidate research gaps from retrieved evidence chunks.

    Dependencies (injected)
    -----------------------
    llm_provider : LLMProvider
        Used to prompt an LLM to analyse chunks and identify gaps.

    Implementation plan (not yet implemented)
    -----------------------------------------
    1. Build a structured prompt from the query and retrieved chunks.
    2. Call the LLM to identify gap statements and classify each gap type.
    3. Parse the LLM response into CandidateGap domain objects.
    4. Apply the structured representation step (representation.py).

    Status: SKELETON.
    """

    def __init__(self, llm_provider: object) -> None:
        self._llm = llm_provider

    def detect(
        self,
        query: ResearchQuery,
        chunks: list[DocumentChunk],
    ) -> list[CandidateGap]:
        """
        Detect candidate gaps.  NOT YET IMPLEMENTED.
        Returns an empty list until the implementation is added.
        """
        logger.warning("GapDetector.detect is not yet implemented.")
        return []
