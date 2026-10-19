"""
tests/unit/test_gap_representation.py
---------------------------------------
Unit tests for GapRepresentationBuilder.  LLM calls are mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.exceptions import GapDetectionError, ProviderError
from app.domain.models.candidate_gap import StructuredGapRepresentation
from app.gap_detection.representation import GapRepresentationBuilder
from app.gap_detection.schemas import GapRepresentationLLMOutput

STATEMENT = "No benchmark evaluates GNN-based CAN intrusion detection on real vehicles."


def _builder(return_value=None, raises=None) -> tuple[GapRepresentationBuilder, MagicMock]:
    llm = MagicMock()
    if raises is not None:
        llm.complete_structured.side_effect = raises
    else:
        llm.complete_structured.return_value = return_value
    return GapRepresentationBuilder(llm), llm


class TestFromLLMOutput:
    def test_maps_all_fields(self):
        output = GapRepresentationLLMOutput(
            domain="Automotive security",
            population="real vehicles",
            methodology="GNN",
            outcome="intrusion detection",
            temporal_scope="post-2020",
            geographic_scope="EU",
            key_concepts=["CAN bus", "GNN"],
            synonyms=["Controller Area Network"],
        )
        rep = GapRepresentationBuilder.from_llm_output(STATEMENT, output)
        assert isinstance(rep, StructuredGapRepresentation)
        assert rep.gap_statement == STATEMENT
        assert rep.methodology == "GNN"
        assert rep.temporal_scope == "post-2020"
        assert rep.geographic_scope == "EU"
        assert rep.key_concepts == ["CAN bus", "GNN"]

    def test_case_insensitive_dedupe(self):
        output = GapRepresentationLLMOutput(key_concepts=["GNN", "gnn", "CAN"])
        rep = GapRepresentationBuilder.from_llm_output(STATEMENT, output)
        assert rep.key_concepts == ["GNN", "CAN"]


class TestBuild:
    def test_build_calls_llm_and_maps(self):
        builder, llm = _builder({"domain": "Automotive security", "key_concepts": ["CAN bus"]})
        rep = builder.build(STATEMENT)
        assert rep.domain == "Automotive security"
        assert rep.key_concepts == ["CAN bus"]
        assert rep.population is None
        prompt, schema = llm.complete_structured.call_args.args
        assert STATEMENT in prompt
        assert "key_concepts" in schema["properties"]

    def test_statement_is_stripped(self):
        builder, _ = _builder({})
        assert builder.build(f"  {STATEMENT}  ").gap_statement == STATEMENT

    def test_blank_statement_raises_without_llm_call(self):
        builder, llm = _builder({})
        with pytest.raises(GapDetectionError):
            builder.build("   ")
        llm.complete_structured.assert_not_called()

    def test_no_llm_configured_raises(self):
        with pytest.raises(GapDetectionError):
            GapRepresentationBuilder(None).build(STATEMENT)

    def test_provider_error_wrapped(self):
        builder, _ = _builder(raises=ProviderError("rate limited"))
        with pytest.raises(GapDetectionError):
            builder.build(STATEMENT)

    def test_non_dict_response_raises(self):
        builder, _ = _builder("text")
        with pytest.raises(GapDetectionError):
            builder.build(STATEMENT)

    def test_bare_string_list_field_coerced(self):
        """M2 regression: a bare string for a list field no longer fails the call."""
        builder, _ = _builder({"key_concepts": "CAN bus", "synonyms": {"bad": "type"}})
        rep = builder.build(STATEMENT)
        assert rep.key_concepts == ["CAN bus"]
        assert rep.synonyms == []

    def test_provider_error_cause_preserved(self):
        """M3 regression: the provider's cause reaches the raised error."""
        builder, _ = _builder(
            raises=ProviderError("Error code: 429 - rate limit exceeded", detail="429 Too Many Requests")
        )
        with pytest.raises(GapDetectionError) as exc_info:
            builder.build(STATEMENT)
        assert "429" in str(exc_info.value)
        assert "gap representation" in str(exc_info.value)
        assert exc_info.value.detail == "429 Too Many Requests"
