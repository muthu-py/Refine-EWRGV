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


def reason_code(classification: GapClassification, components: dict, detectors: list[str]) -> str:
    """
    Why a gap got its class, for the report:

    SUPPORTED              VALID_GAP: support clearly outweighs counter evidence
    ADDRESSED_EXTERNALLY   counter / partial work found, or high external coverage
    CORPUS_ARTEFACT        sparse in the run corpus (D2 / D4) but well covered externally
    CONTESTED              counter mass at least half of support
    INSUFFICIENT_EVIDENCE  too little relevant evidence (a confidence cap applied)
    WEAK_SUPPORT           otherwise
    """
    caps = components.get("caps_applied") or []
    counts = components.get("evidence_counts", {})
    cov = components.get("external_coverage")
    structural = bool(set(detectors) & {"combination_sparsity", "topic_density"}) and "explicit_statements" not in detectors
    if any(c.startswith("insufficient") for c in caps):
        return "INSUFFICIENT_EVIDENCE"
    if structural and cov is not None and cov >= 0.15:
        return "CORPUS_ARTEFACT"
    if classification == GapClassification.VALID_GAP:
        return "SUPPORTED"
    if components.get("coverage_penalty", 0) >= 0.5 or (counts.get("Counter", 0) + counts.get("Contradictory", 0)) >= 2:
        return "ADDRESSED_EXTERNALLY"
    if components.get("counter", 0) >= 0.5 * components.get("support", 1):
        return "CONTESTED"
    return "WEAK_SUPPORT"
