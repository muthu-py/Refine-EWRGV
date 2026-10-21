"""
app/scoring/confidence.py
--------------------------
EWRGV Confidence Assessment: a confidence in [0, 1] that a candidate gap is
genuine, from the candidate's own evidence (GapScore), the classified
validation evidence and coverage.

    support  = 2 * GapScore + sum strength(SUPPORTING)
    counter  = 1.2 * sum strength(COUNTER) + 0.6 * sum strength(PARTIAL)
             + 1.2 * sum strength(CONTRADICTORY)                (conservative w_c = 1.2)
    base     = support / (support + counter)

    coverage_penalty = clip((cov_ext - 0.02) / (0.15 - 0.02), 0, 1)   (external coverage, coverage/analyzer.py)
    confidence       = base * (1 - 0.6 * coverage_penalty)

Caps ("sparse != gap"):

    * fewer than 3 relevant evidence items, or retrieval coverage < 0.25
                                  -> confidence <= 0.39 (insufficient evidence: never VALID or UNCERTAIN)
    * corpus_adequacy < 0.4       -> confidence <= 0.69 (an inadequate evidence base cannot yield VALID_GAP)

Thresholds (settings): VALID_GAP >= 0.70, UNCERTAIN_GAP >= 0.40, else UNSUPPORTED_GAP.
"""

from __future__ import annotations

from typing import Optional

from app.core.logging import get_logger
from app.domain.models.validation import CoverageAssessment

logger = get_logger(__name__)

COUNTER_WEIGHT = 1.2
PARTIAL_WEIGHT = 0.6
CONTRADICTORY_WEIGHT = 1.2
GAPSCORE_WEIGHT = 2.0
COVERAGE_LOW, COVERAGE_HIGH = 0.02, 0.15
COVERAGE_PENALTY = 0.6
MIN_RELEVANT_ITEMS = 3
MIN_RETRIEVAL_COVERAGE = 0.25
INSUFFICIENT_CAP = 0.39
ADEQUACY_THRESHOLD = 0.4
ADEQUACY_CAP = 0.69


class EWRGVConfidenceScorer:
    def score(self, gap_score: float, items: list[dict], coverage: CoverageAssessment,
              external_coverage: Optional[dict] = None) -> dict:
        mass = {"Supporting": 0.0, "Counter": 0.0, "Partial": 0.0, "Contradictory": 0.0}
        counts = {"Supporting": 0, "Counter": 0, "Partial": 0, "Contradictory": 0, "Insufficient": 0}
        for it in items:
            counts[it["evidence_type"]] = counts.get(it["evidence_type"], 0) + 1
            if it["evidence_type"] in mass:
                mass[it["evidence_type"]] += it["strength"]
        support = GAPSCORE_WEIGHT * gap_score + mass["Supporting"]
        counter = COUNTER_WEIGHT * mass["Counter"] + PARTIAL_WEIGHT * mass["Partial"] + CONTRADICTORY_WEIGHT * mass["Contradictory"]
        base = support / (support + counter) if support + counter > 0 else 0.0

        cov = external_coverage.get("coverage") if external_coverage and external_coverage.get("available") else None
        penalty = 0.0 if cov is None else min(1.0, max(0.0, (cov - COVERAGE_LOW) / (COVERAGE_HIGH - COVERAGE_LOW)))
        confidence = base * (1 - COVERAGE_PENALTY * penalty)

        caps = []
        relevant = sum(v for k, v in counts.items() if k != "Insufficient")
        if relevant < MIN_RELEVANT_ITEMS or (coverage.retrieval_coverage is not None and coverage.retrieval_coverage < MIN_RETRIEVAL_COVERAGE):
            if confidence > INSUFFICIENT_CAP:
                caps.append(f"insufficient evidence: {relevant} relevant items, retrieval coverage {coverage.retrieval_coverage}")
            confidence = min(confidence, INSUFFICIENT_CAP)
        if coverage.corpus_adequacy is not None and coverage.corpus_adequacy < ADEQUACY_THRESHOLD:
            if confidence > ADEQUACY_CAP:
                caps.append(f"corpus adequacy {coverage.corpus_adequacy:.2f} < {ADEQUACY_THRESHOLD}")
            confidence = min(confidence, ADEQUACY_CAP)
        return {
            "confidence": round(confidence, 4),
            "components": {
                "support": round(support, 4), "counter": round(counter, 4), "base": round(base, 4),
                "gap_score": gap_score, "evidence_mass": {k: round(v, 4) for k, v in mass.items()},
                "evidence_counts": counts, "external_coverage": cov, "coverage_penalty": round(penalty, 4),
                "caps_applied": caps,
            },
        }
