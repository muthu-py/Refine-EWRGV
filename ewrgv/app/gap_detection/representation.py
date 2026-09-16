"""
app/gap_detection/representation.py
-------------------------------------
Structured gap representation.

Takes a natural-language gap statement and produces a StructuredGapRepresentation
that decomposes the gap into addressable components (domain, population,
methodology, outcome, key concepts, synonyms).

This structured form is consumed by the EWRGV validation module to generate
targeted verification queries.

Status: SKELETON – not implemented.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.models.candidate_gap import StructuredGapRepresentation

logger = get_logger(__name__)


class GapRepresentationBuilder:
    """
    Produces a StructuredGapRepresentation from a natural-language gap statement.

    Depends on an LLMProvider to extract structured fields.

    Status: SKELETON – not implemented.
    """

    def __init__(self, llm_provider: object) -> None:
        self._llm = llm_provider

    def build(self, gap_statement: str) -> StructuredGapRepresentation:
        """
        Parse the gap statement into a structured representation.
        NOT YET IMPLEMENTED.
        """
        logger.warning("GapRepresentationBuilder.build is not yet implemented.")
        return StructuredGapRepresentation(gap_statement=gap_statement)
