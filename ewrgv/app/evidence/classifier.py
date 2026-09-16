"""
app/evidence/classifier.py
---------------------------
Evidence classifier: assigns an EvidenceType to a DocumentChunk relative to a gap.

Satisfies the EvidenceClassifier interface (app.domain.interfaces).

Status: SKELETON – not implemented.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.enums import EvidenceType
from app.domain.models.candidate_gap import CandidateGap
from app.domain.models.paper import DocumentChunk

logger = get_logger(__name__)


class LLMEvidenceClassifier:
    """
    Classifies evidence using an LLM to determine:
    SUPPORTING | COUNTER | PARTIAL | CONTRADICTORY | INSUFFICIENT

    A separate classification call is made per (gap, chunk) pair.

    Status: SKELETON – not implemented.
    """

    def __init__(self, llm_provider: object) -> None:
        self._llm = llm_provider

    def classify(self, gap: CandidateGap, chunk: DocumentChunk) -> EvidenceType:
        """Classify a chunk relative to the gap. NOT YET IMPLEMENTED."""
        logger.warning("LLMEvidenceClassifier.classify is not yet implemented.")
        return EvidenceType.INSUFFICIENT
