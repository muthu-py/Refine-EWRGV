"""
app/query_understanding/schemas.py
------------------------------------
LLM-layer transport schema for the Query Understanding & Expansion output.

QueryUnderstandingLLMOutput is the Pydantic model that:
  1. Defines the exact JSON structure the LLM must produce.
  2. Validates the raw LLM response before any domain mapping occurs.
  3. Provides the JSON Schema dict sent to the LLM provider.

Separation from the domain model
---------------------------------
This schema lives in the infrastructure/service layer, not in the domain.
The service (QueryUnderstandingService) maps from this schema to the
domain model (QueryUnderstanding) after validation.

This separation means:
  - The LLM contract can change without breaking the domain model, and
  - The domain model can evolve without requiring prompt or schema changes.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class QueryUnderstandingLLMOutput(BaseModel):
    """
    Structured output returned by the LLM for a query understanding request.

    All list fields are optional from the LLM's perspective (it may return
    null); the validators normalise null → empty list.

    The ``expanded_queries`` field must be non-empty and each entry must be a
    non-whitespace string.
    """

    domain: Optional[str] = None
    problem: Optional[str] = None
    task: Optional[str] = None
    methods: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    research_intent: Optional[str] = None
    terminology_variants: list[str] = Field(default_factory=list)
    method_variants: list[str] = Field(default_factory=list)
    expanded_queries: list[str] = Field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #

    @field_validator(
        "methods",
        "concepts",
        "entities",
        "datasets",
        "metrics",
        "constraints",
        "terminology_variants",
        "method_variants",
        "expanded_queries",
        mode="before",
    )
    @classmethod
    def coerce_none_list_to_empty(cls, v: object) -> list:
        """Treat an explicit null from the LLM as an empty list."""
        if v is None:
            return []
        return v

    @field_validator("expanded_queries", mode="after")
    @classmethod
    def validate_expanded_queries(cls, v: list[str]) -> list[str]:
        """
        Each expanded query must be a non-empty, non-whitespace string.
        Silently drop blank entries; if nothing remains, return empty.
        """
        cleaned = [q.strip() for q in v if isinstance(q, str) and q.strip()]
        return cleaned

    # ------------------------------------------------------------------ #
    # JSON Schema for LLM provider
    # ------------------------------------------------------------------ #

    @classmethod
    def json_schema_for_llm(cls) -> dict:
        """
        Return the JSON Schema dict to send to the LLM provider as context.

        Using a hand-crafted schema keeps it concise and provider-friendly;
        the full Pydantic-generated schema can include internal Pydantic
        metadata that confuses some models.
        """
        return {
            "type": "object",
            "properties": {
                "domain": {"type": ["string", "null"]},
                "problem": {"type": ["string", "null"]},
                "task": {"type": ["string", "null"]},
                "methods": {"type": "array", "items": {"type": "string"}},
                "concepts": {"type": "array", "items": {"type": "string"}},
                "entities": {"type": "array", "items": {"type": "string"}},
                "datasets": {"type": "array", "items": {"type": "string"}},
                "metrics": {"type": "array", "items": {"type": "string"}},
                "constraints": {"type": "array", "items": {"type": "string"}},
                "research_intent": {"type": ["string", "null"]},
                "terminology_variants": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "method_variants": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "expanded_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "description": (
                        "Diverse keyword-style search queries for "
                        "academic literature databases."
                    ),
                },
            },
            "required": [
                "domain",
                "problem",
                "task",
                "methods",
                "concepts",
                "entities",
                "datasets",
                "metrics",
                "constraints",
                "research_intent",
                "terminology_variants",
                "method_variants",
                "expanded_queries",
            ],
        }
