"""
app/validation/query_generation.py
------------------------------------
Generates targeted verification queries from a CandidateGap's structured
representation.

Each generated query is used by one of the four EWRGV validation searches:
    direct, terminology, method_variation, counter.

Status: SKELETON – not implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.core.logging import get_logger
from app.domain.models.candidate_gap import CandidateGap

logger = get_logger(__name__)


@dataclass
class VerificationQueries:
    """
    Typed container for queries generated for each EWRGV search type.
    """

    direct: list[str] = field(default_factory=list)
    terminology: list[str] = field(default_factory=list)
    method_variation: list[str] = field(default_factory=list)
    counter: list[str] = field(default_factory=list)


class VerificationQueryGenerator:
    """
    Generates verification queries from a CandidateGap.

    Strategy
    --------
    * direct       – queries that directly test the gap claim
    * terminology  – queries using synonyms and related terms
    * method_variation – queries targeting different methodologies
    * counter      – queries specifically designed to find counter-evidence

    Status: SKELETON – not implemented.
    """

    def __init__(self, llm_provider: object) -> None:
        self._llm = llm_provider

    def generate(self, gap: CandidateGap) -> VerificationQueries:
        """Generate all verification query sets. NOT YET IMPLEMENTED."""
        logger.warning("VerificationQueryGenerator.generate is not yet implemented.")
        return VerificationQueries()
