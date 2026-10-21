"""
app/gap_detection/representation.py
-------------------------------------
Structured gap representation.

Takes a natural-language gap statement and produces a StructuredGapRepresentation
that decomposes the gap into addressable components (domain, population,
methodology, outcome, key concepts, synonyms).

This structured form is consumed by the EWRGV validation module to generate
targeted verification queries.

Two entry points
----------------
* ``GapRepresentationBuilder.from_llm_output()`` — pure mapping from an
  already-validated LLM output.  Used by GapDetector, whose single combined
  LLM call already returns the representation fields for every candidate.
* ``GapRepresentationBuilder.build()`` — makes its own LLM call for a
  standalone gap statement (e.g. a gap supplied by a user rather than by
  the detector).
"""

from __future__ import annotations

import pydantic

from app.core.exceptions import GapDetectionError
from app.core.logging import get_logger
from app.domain.interfaces import LLMProvider
from app.domain.models.candidate_gap import StructuredGapRepresentation
from app.gap_detection.llm_call import call_structured_llm
from app.gap_detection.prompts import build_gap_representation_prompt
from app.gap_detection.schemas import GapRepresentationLLMOutput

logger = get_logger(__name__)


def _dedupe_preserving_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = item.casefold()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


class GapRepresentationBuilder:
    """
    Produces a StructuredGapRepresentation from a natural-language gap statement.

    Parameters
    ----------
    llm_provider:
        Any object satisfying the LLMProvider protocol.  Only required by
        ``build()``; ``from_llm_output()`` makes no LLM call.
    """

    def __init__(self, llm_provider: LLMProvider | None) -> None:
        self._llm = llm_provider

    @staticmethod
    def from_llm_output(
        gap_statement: str,
        output: GapRepresentationLLMOutput,
    ) -> StructuredGapRepresentation:
        """Map validated LLM representation fields to the domain model."""
        return StructuredGapRepresentation(
            gap_statement=gap_statement,
            domain=output.domain,
            population=output.population,
            methodology=output.methodology,
            outcome=output.outcome,
            temporal_scope=output.temporal_scope,
            geographic_scope=output.geographic_scope,
            key_concepts=_dedupe_preserving_order(output.key_concepts),
            synonyms=_dedupe_preserving_order(output.synonyms),
        )

    def build(self, gap_statement: str) -> StructuredGapRepresentation:
        """
        Parse the gap statement into a structured representation via the LLM.

        Raises
        ------
        GapDetectionError
            If the statement is blank, no LLM provider is configured, the
            provider fails, or the LLM output is invalid.
        """
        statement = (gap_statement or "").strip()
        if not statement:
            raise GapDetectionError("Cannot build a representation for a blank gap statement.")
        if self._llm is None:
            raise GapDetectionError(
                "GapRepresentationBuilder.build() requires an LLM provider."
            )

        raw = call_structured_llm(
            self._llm,
            build_gap_representation_prompt(statement),
            GapRepresentationLLMOutput.json_schema_for_llm(),
            operation="gap representation",
        )
        try:
            output = GapRepresentationLLMOutput.model_validate(raw)
        except pydantic.ValidationError as exc:
            raise GapDetectionError(
                "LLM returned invalid structured output for gap representation.",
                detail=str(exc),
            ) from exc

        return self.from_llm_output(statement, output)
