"""
app/scoring/classification.py
-------------------------------
Gap classification: maps a confidence score to a GapClassification enum.

    confidence >= CONFIDENCE_THRESHOLD_VALID   → VALID_GAP
    confidence >= CONFIDENCE_THRESHOLD_UNCERTAIN → UNCERTAIN_GAP
    confidence <  CONFIDENCE_THRESHOLD_UNCERTAIN → UNSUPPORTED_GAP

Thresholds are read from settings and may be overridden per run.
"""

from __future__ import annotations

from app.domain.enums import GapClassification


def classify_by_confidence(
    confidence: float,
    threshold_valid: float = 0.70,
    threshold_uncertain: float = 0.40,
) -> GapClassification:
    """
    Map a confidence score to a GapClassification.

    Parameters
    ----------
    confidence:
        Score in [0.0, 1.0] from EWRGVConfidenceScorer.
    threshold_valid:
        Minimum confidence to be classified as VALID_GAP.
    threshold_uncertain:
        Minimum confidence to be classified as UNCERTAIN_GAP.

    Returns
    -------
    GapClassification
    """
    if confidence >= threshold_valid:
        return GapClassification.VALID_GAP
    elif confidence >= threshold_uncertain:
        return GapClassification.UNCERTAIN_GAP
    else:
        return GapClassification.UNSUPPORTED_GAP
