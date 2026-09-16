"""
app/domain/models/query_understanding.py
-----------------------------------------
Domain model for the structured understanding produced by the
Query Understanding & Expansion module (Phase 2.1).

QueryUnderstanding is a pure domain object — it carries no infrastructure
concerns and no LLM-provider-specific types.  It is populated by the
QueryUnderstandingService and attached to ResearchQuery.understanding.

Design note
-----------
This model is intentionally *separate* from the LLM schema
(app.query_understanding.schemas.QueryUnderstandingLLMOutput).
The LLM schema is the transport/contract layer; this model is the
domain representation.  The service maps between them.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class QueryUnderstanding(BaseModel):
    """
    Structured decomposition of a research question.

    Produced once by the QueryUnderstandingService and attached to
    ResearchQuery.understanding.  All list fields default to empty;
    all string fields default to None — the LLM only populates fields
    it can reasonably infer from the question.

    Attributes
    ----------
    domain:
        The high-level research area (e.g. "Automotive Cybersecurity").
    problem:
        The central problem being studied (e.g. "CAN bus attack detection").
    task:
        The ML / research task (e.g. "intrusion detection").
    methods:
        Methods or techniques mentioned or strongly implied.
    concepts:
        Key concepts important for literature search.
    entities:
        Named entities (organisations, products, standards).
    datasets:
        Benchmark / evaluation datasets explicitly mentioned.
    metrics:
        Evaluation metrics explicitly mentioned or strongly implied.
    constraints:
        Scope constraints: geography, population, timeframe, modality, etc.
    research_intent:
        High-level intent of the question (e.g. "method exploration",
        "survey", "comparative study").
    terminology_variants:
        Alternative terms/abbreviations useful for broadening search.
    method_variants:
        Alternative method names useful for broadening search.
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

    @field_validator(
        "methods",
        "concepts",
        "entities",
        "datasets",
        "metrics",
        "constraints",
        "terminology_variants",
        "method_variants",
        mode="before",
    )
    @classmethod
    def coerce_none_list_to_empty(cls, v: object) -> list:
        """Treat an explicit None from the LLM as an empty list."""
        if v is None:
            return []
        return v
