"""
app/evidence/aggregator.py
---------------------------
Evidence aggregator: merges evidence collected across multiple validation
search types into a de-duplicated, weighted evidence list.

Status: SKELETON – not implemented.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.models.evidence import Evidence

logger = get_logger(__name__)


class EvidenceAggregator:
    """
    Combines evidence from direct, terminology, method, and counter searches.

    Responsibilities
    ----------------
    * De-duplicate evidence items referencing the same passage.
    * Weight evidence by search type and relevance.
    * Produce a final evidence list for coverage analysis and scoring.

    Status: SKELETON – not implemented.
    """

    def aggregate(self, evidence_items: list[Evidence]) -> list[Evidence]:
        """Aggregate and de-duplicate evidence. NOT YET IMPLEMENTED."""
        logger.warning("EvidenceAggregator.aggregate is not yet implemented.")
        return evidence_items
