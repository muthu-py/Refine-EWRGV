"""
app/explanation/evidence_chain.py
----------------------------------
Evidence chain builder.

Constructs a step-by-step EvidenceChainEntry list that traces the reasoning
from the CandidateGap through each validation search type, evidence
classification, coverage, and confidence to the final classification.

Status: SKELETON – not implemented.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.models.evidence import Evidence
from app.domain.models.validation import EvidenceChainEntry

logger = get_logger(__name__)


def build_evidence_chain(evidence: list[Evidence]) -> list[EvidenceChainEntry]:
    """
    Build a structured evidence chain from the collected evidence.
    NOT YET IMPLEMENTED – returns an empty chain.
    """
    logger.warning("build_evidence_chain is not yet implemented.")
    return []
