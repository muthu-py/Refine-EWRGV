"""
app/evidence/strength.py
--------------------------
Evidence strength assessment.

Determines how strongly a piece of evidence supports or counters a gap claim,
independent of evidence type classification.

Status: SKELETON – placeholder module.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.models.evidence import Evidence

logger = get_logger(__name__)


def assess_strength(evidence: Evidence) -> float:
    """
    Compute a strength score for an evidence item.

    Returns a score in [0.0, 1.0].
    NOT YET IMPLEMENTED — returns 0.0.
    """
    logger.warning("assess_strength is not yet implemented.")
    return 0.0
