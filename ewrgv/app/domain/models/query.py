"""
app/domain/models/query.py
--------------------------
Domain model for a research query submitted to the EWRGV pipeline.

ResearchQuery is created at the entry point of the pipeline and is
progressively enriched as query-understanding and query-expansion run.
It is intentionally kept free of any infrastructure concerns.
"""

from __future__ import annotations

from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.domain.enums import QueryType


class ResearchQuery(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    """
    Represents the research question as it flows through the pipeline.

    Attributes
    ----------
    query_id:
        Unique identifier for this query session.
    query:
        The raw research question provided by the user.
    query_type:
        High-level classification determined during query understanding.
    expanded_queries:
        List of reformulated / synonym-expanded queries generated from the
        original query during the query-expansion step.
    metadata:
        Optional free-form metadata (e.g., domain hints, language).
    """

    query_id: str = Field(default_factory=lambda: str(uuid4()))
    query: str
    query_type: QueryType = QueryType.UNKNOWN
    expanded_queries: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

