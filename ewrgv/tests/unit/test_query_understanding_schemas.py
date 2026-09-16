"""
tests/unit/test_query_understanding_schemas.py
------------------------------------------------
Unit tests for the LLM transport schema (QueryUnderstandingLLMOutput)
and the domain model (QueryUnderstanding).

These tests run fully offline — no LLM calls, no API keys.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from app.domain.models.query_understanding import QueryUnderstanding
from app.query_understanding.schemas import QueryUnderstandingLLMOutput


# ======================================================================
# QueryUnderstandingLLMOutput — validation tests
# ======================================================================


class TestQueryUnderstandingLLMOutput:
    """Tests for the LLM output schema / transport layer."""

    def _valid_payload(self) -> dict:
        """Return a minimal valid LLM payload."""
        return {
            "domain": "Automotive Cybersecurity",
            "problem": "CAN bus attack detection",
            "task": "intrusion detection",
            "methods": ["Graph Neural Networks"],
            "concepts": ["CAN bus", "autonomous vehicles"],
            "entities": [],
            "datasets": [],
            "metrics": [],
            "constraints": [],
            "research_intent": "method exploration",
            "terminology_variants": ["Controller Area Network"],
            "method_variants": ["GNN", "GCN"],
            "expanded_queries": [
                "GNN CAN bus intrusion detection",
                "graph neural networks automotive cybersecurity",
            ],
        }

    def test_valid_payload_parses_correctly(self):
        payload = self._valid_payload()
        output = QueryUnderstandingLLMOutput.model_validate(payload)
        assert output.domain == "Automotive Cybersecurity"
        assert output.problem == "CAN bus attack detection"
        assert output.task == "intrusion detection"
        assert output.methods == ["Graph Neural Networks"]
        assert output.concepts == ["CAN bus", "autonomous vehicles"]
        assert output.research_intent == "method exploration"
        assert output.terminology_variants == ["Controller Area Network"]
        assert output.method_variants == ["GNN", "GCN"]
        assert len(output.expanded_queries) == 2

    def test_null_list_fields_coerced_to_empty(self):
        payload = self._valid_payload()
        payload["methods"] = None
        payload["concepts"] = None
        payload["entities"] = None
        output = QueryUnderstandingLLMOutput.model_validate(payload)
        assert output.methods == []
        assert output.concepts == []
        assert output.entities == []

    def test_null_optional_string_fields_are_none(self):
        payload = self._valid_payload()
        payload["domain"] = None
        payload["problem"] = None
        payload["research_intent"] = None
        output = QueryUnderstandingLLMOutput.model_validate(payload)
        assert output.domain is None
        assert output.problem is None
        assert output.research_intent is None

    def test_expanded_queries_whitespace_entries_stripped(self):
        payload = self._valid_payload()
        payload["expanded_queries"] = [
            "valid query",
            "  ",        # blank — should be removed
            "",          # empty — should be removed
            "  another valid  ",
        ]
        output = QueryUnderstandingLLMOutput.model_validate(payload)
        assert output.expanded_queries == ["valid query", "another valid"]

    def test_all_expanded_queries_blank_returns_empty(self):
        payload = self._valid_payload()
        payload["expanded_queries"] = ["  ", "", "   "]
        output = QueryUnderstandingLLMOutput.model_validate(payload)
        assert output.expanded_queries == []

    def test_missing_optional_fields_use_defaults(self):
        """Missing optional list fields should default to empty lists."""
        minimal = {
            "domain": None,
            "problem": None,
            "task": None,
            "research_intent": None,
            "expanded_queries": ["some query"],
        }
        output = QueryUnderstandingLLMOutput.model_validate(minimal)
        assert output.methods == []
        assert output.concepts == []
        assert output.entities == []
        assert output.datasets == []
        assert output.metrics == []
        assert output.constraints == []
        assert output.terminology_variants == []
        assert output.method_variants == []

    def test_json_schema_for_llm_has_required_keys(self):
        schema = QueryUnderstandingLLMOutput.json_schema_for_llm()
        assert schema["type"] == "object"
        required_fields = {
            "domain", "problem", "task", "methods", "concepts", "entities",
            "datasets", "metrics", "constraints", "research_intent",
            "terminology_variants", "method_variants", "expanded_queries",
        }
        assert set(schema["required"]) == required_fields
        assert "expanded_queries" in schema["properties"]

    def test_expanded_queries_null_coerced_to_empty(self):
        payload = self._valid_payload()
        payload["expanded_queries"] = None
        output = QueryUnderstandingLLMOutput.model_validate(payload)
        assert output.expanded_queries == []


# ======================================================================
# QueryUnderstanding — domain model tests
# ======================================================================


class TestQueryUnderstanding:
    """Tests for the domain model."""

    def test_defaults_are_safe(self):
        qu = QueryUnderstanding()
        assert qu.domain is None
        assert qu.problem is None
        assert qu.task is None
        assert qu.methods == []
        assert qu.concepts == []
        assert qu.entities == []
        assert qu.datasets == []
        assert qu.metrics == []
        assert qu.constraints == []
        assert qu.research_intent is None
        assert qu.terminology_variants == []
        assert qu.method_variants == []

    def test_fully_populated(self):
        qu = QueryUnderstanding(
            domain="NLP",
            problem="Bias in language models",
            task="bias detection",
            methods=["BERT", "GPT"],
            concepts=["language model", "bias"],
            entities=["HuggingFace"],
            datasets=["WinoBias"],
            metrics=["F1-score"],
            constraints=["English language"],
            research_intent="survey",
            terminology_variants=["LLM bias", "model fairness"],
            method_variants=["RoBERTa", "DeBERTa"],
        )
        assert qu.domain == "NLP"
        assert qu.methods == ["BERT", "GPT"]
        assert qu.research_intent == "survey"

    def test_null_list_fields_normalised_to_empty(self):
        qu = QueryUnderstanding(methods=None, concepts=None)  # type: ignore[arg-type]
        assert qu.methods == []
        assert qu.concepts == []
