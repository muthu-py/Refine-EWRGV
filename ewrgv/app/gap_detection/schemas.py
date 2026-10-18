"""
app/gap_detection/schemas.py
------------------------------
LLM-layer transport schemas for candidate gap detection.

These Pydantic models:
  1. Define the exact JSON structure the LLM must produce.
  2. Validate the raw LLM response before any domain mapping occurs.
  3. Provide the JSON Schema dicts sent to the LLM provider.

Separation from the domain model
---------------------------------
These schemas live in the service layer, not in the domain.  The detector
(GapDetector) and representation builder (GapRepresentationBuilder) map
from these schemas to the domain models (CandidateGap,
StructuredGapRepresentation) after validation.  This mirrors the pattern
used by app/query_understanding/schemas.py.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from app.domain.enums import GapType


def _clean_string_list(v: object) -> list:
    """
    Leniently coerce an LLM list field to ``list[str]``.

    null → [], a bare string → [string] (a common LLM slip), any other
    non-list value → [].  Blank and non-string entries are dropped.
    """
    if v is None:
        return []
    if isinstance(v, str):
        return [v.strip()] if v.strip() else []
    if not isinstance(v, list):
        return []
    return [s.strip() for s in v if isinstance(s, str) and s.strip()]


def _clean_optional_string(v: object) -> str | None:
    """Normalise blank strings and non-string values to None."""
    if not isinstance(v, str):
        return None
    return v.strip() or None


_REPRESENTATION_SCHEMA_PROPERTIES: dict = {
    "domain": {"type": ["string", "null"]},
    "population": {"type": ["string", "null"]},
    "methodology": {"type": ["string", "null"]},
    "outcome": {"type": ["string", "null"]},
    "temporal_scope": {"type": ["string", "null"]},
    "geographic_scope": {"type": ["string", "null"]},
    "key_concepts": {"type": "array", "items": {"type": "string"}},
    "synonyms": {"type": "array", "items": {"type": "string"}},
}


class GapRepresentationLLMOutput(BaseModel):
    """
    Structured decomposition of a gap statement as returned by the LLM.

    Shared by the combined detection call (embedded in each candidate)
    and by the standalone representation call.
    """

    domain: Optional[str] = None
    population: Optional[str] = None
    methodology: Optional[str] = None
    outcome: Optional[str] = None
    temporal_scope: Optional[str] = None
    geographic_scope: Optional[str] = None
    key_concepts: list[str] = Field(default_factory=list)
    synonyms: list[str] = Field(default_factory=list)

    @field_validator("key_concepts", "synonyms", mode="before")
    @classmethod
    def coerce_string_lists(cls, v: object) -> list:
        return _clean_string_list(v)

    @field_validator(
        "domain",
        "population",
        "methodology",
        "outcome",
        "temporal_scope",
        "geographic_scope",
        mode="before",
    )
    @classmethod
    def coerce_blank_strings(cls, v: object) -> object:
        return _clean_optional_string(v)

    @classmethod
    def json_schema_for_llm(cls) -> dict:
        """Return the hand-crafted JSON Schema for the standalone call."""
        return {
            "type": "object",
            "properties": dict(_REPRESENTATION_SCHEMA_PROPERTIES),
            "required": list(_REPRESENTATION_SCHEMA_PROPERTIES),
        }


class CandidateGapLLMOutput(GapRepresentationLLMOutput):
    """
    A single candidate gap as returned by the detection LLM call.

    Combines the gap statement, its taxonomy labels, the chunk IDs that
    ground it, the detector's initial confidence, and the structured
    representation fields (inherited).

    ``gap_type`` and ``secondary_gap_types`` are kept as raw strings here;
    mapping to the GapType enum (with an OTHER fallback) happens in the
    detector so that one unrecognised label does not invalidate the
    whole response.
    """

    gap_statement: str
    gap_type: Optional[str] = None
    secondary_gap_types: list[str] = Field(default_factory=list)
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    contradicting_chunk_ids: list[str] = Field(default_factory=list)
    initial_confidence: float = 0.0
    rationale: Optional[str] = None

    @field_validator("gap_statement", mode="before")
    @classmethod
    def strip_statement(cls, v: object) -> object:
        return v.strip() if isinstance(v, str) else v

    @field_validator(
        "secondary_gap_types", "supporting_chunk_ids", "contradicting_chunk_ids", mode="before"
    )
    @classmethod
    def coerce_raw_lists(cls, v: object) -> list:
        return _clean_string_list(v)

    @field_validator("gap_type", "rationale", mode="before")
    @classmethod
    def coerce_optional_text(cls, v: object) -> object:
        return _clean_optional_string(v)

    @field_validator("initial_confidence", mode="before")
    @classmethod
    def coerce_confidence(cls, v: object) -> float:
        """Coerce to float and clamp to [0.0, 1.0]; unparseable → 0.0."""
        try:
            value = float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return 0.0
        if value != value:  # NaN
            return 0.0
        return min(max(value, 0.0), 1.0)


class GapDetectionLLMOutput(BaseModel):
    """
    Structured output returned by the LLM for a gap detection request.

    Top-level validation stays strict: the response must be an object, and
    ``candidate_gaps`` must be a list (null or missing is treated as empty).

    Candidate items are validated individually.  An item that is not an
    object, has a blank ``gap_statement``, or otherwise fails validation is
    dropped and described in ``dropped_candidates`` so that one malformed
    candidate does not discard the valid ones.  An empty ``candidate_gaps``
    list is valid (the evidence may not indicate any gap).
    """

    candidate_gaps: list[CandidateGapLLMOutput] = Field(default_factory=list)
    dropped_candidates: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def validate_candidates_individually(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data  # let Pydantic report the top-level type error
        items = data.get("candidate_gaps")
        if items is None:
            return {**data, "candidate_gaps": [], "dropped_candidates": []}
        if not isinstance(items, list):
            return data  # strict: a non-list candidate_gaps is an error

        valid: list[CandidateGapLLMOutput] = []
        dropped: list[str] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                dropped.append(f"item {index}: not an object ({type(item).__name__})")
                continue
            statement = item.get("gap_statement")
            if not (isinstance(statement, str) and statement.strip()):
                dropped.append(f"item {index}: missing or blank gap_statement")
                continue
            try:
                valid.append(CandidateGapLLMOutput.model_validate(item))
            except ValidationError as exc:
                first = exc.errors()[0]
                location = ".".join(str(part) for part in first.get("loc", ()))
                dropped.append(f"item {index}: {location or 'item'}: {first.get('msg')}")
        return {**data, "candidate_gaps": valid, "dropped_candidates": dropped}

    @classmethod
    def json_schema_for_llm(cls) -> dict:
        """Return the hand-crafted JSON Schema for the detection call."""
        candidate_properties = {
            "gap_statement": {"type": "string"},
            "gap_type": {
                "type": "string",
                "enum": [t.value for t in GapType],
            },
            "secondary_gap_types": {
                "type": "array",
                "items": {"type": "string", "enum": [t.value for t in GapType]},
            },
            "supporting_chunk_ids": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": "IDs of the provided passages that ground this gap.",
            },
            "contradicting_chunk_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "IDs of the provided passages suggesting this gap may already "
                    "be addressed (may be empty)."
                ),
            },
            "initial_confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            },
            "rationale": {"type": ["string", "null"]},
            **_REPRESENTATION_SCHEMA_PROPERTIES,
        }
        return {
            "type": "object",
            "properties": {
                "candidate_gaps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": candidate_properties,
                        "required": list(candidate_properties),
                    },
                },
            },
            "required": ["candidate_gaps"],
        }
