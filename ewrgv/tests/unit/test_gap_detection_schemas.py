"""
tests/unit/test_gap_detection_schemas.py
------------------------------------------
Unit tests for the gap detection LLM transport schemas.
"""

from __future__ import annotations

import pydantic
import pytest

from app.domain.enums import GapType
from app.gap_detection.schemas import (
    CandidateGapLLMOutput,
    GapDetectionLLMOutput,
    GapRepresentationLLMOutput,
)


class TestGapRepresentationLLMOutput:
    def test_null_lists_become_empty(self):
        out = GapRepresentationLLMOutput.model_validate({"key_concepts": None, "synonyms": None})
        assert out.key_concepts == []
        assert out.synonyms == []

    def test_blank_list_entries_dropped(self):
        out = GapRepresentationLLMOutput.model_validate(
            {"key_concepts": ["  BERT ", "", "  ", 3, "GPT"]}
        )
        assert out.key_concepts == ["BERT", "GPT"]

    def test_blank_strings_become_none(self):
        out = GapRepresentationLLMOutput.model_validate({"domain": "  ", "population": " x "})
        assert out.domain is None
        assert out.population == "x"

    def test_schema_requires_all_fields(self):
        schema = GapRepresentationLLMOutput.json_schema_for_llm()
        assert set(schema["required"]) == set(schema["properties"])
        assert "key_concepts" in schema["properties"]


class TestCandidateGapLLMOutput:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [(0.5, 0.5), ("0.3", 0.3), (-1, 0.0), (2, 1.0), (None, 0.0), ("high", 0.0),
         (float("nan"), 0.0)],
    )
    def test_confidence_coercion(self, raw, expected):
        out = CandidateGapLLMOutput.model_validate(
            {"gap_statement": "g", "initial_confidence": raw}
        )
        assert out.initial_confidence == pytest.approx(expected)

    def test_statement_required(self):
        with pytest.raises(pydantic.ValidationError):
            CandidateGapLLMOutput.model_validate({"gap_type": "Dataset"})

    def test_inherits_representation_fields(self):
        out = CandidateGapLLMOutput.model_validate(
            {"gap_statement": "g", "methodology": "GNN", "synonyms": None}
        )
        assert out.methodology == "GNN"
        assert out.synonyms == []

    def test_chunk_ids_cleaned(self):
        out = CandidateGapLLMOutput.model_validate(
            {"gap_statement": "g", "supporting_chunk_ids": [" c1 ", "", None]}
        )
        assert out.supporting_chunk_ids == ["c1"]


class TestGapDetectionLLMOutput:
    def test_null_candidates_become_empty(self):
        assert GapDetectionLLMOutput.model_validate({"candidate_gaps": None}).candidate_gaps == []

    def test_missing_candidates_defaults_empty(self):
        assert GapDetectionLLMOutput.model_validate({}).candidate_gaps == []

    def test_blank_statements_dropped(self):
        out = GapDetectionLLMOutput.model_validate(
            {
                "candidate_gaps": [
                    {"gap_statement": "  "},
                    {"gap_statement": None},
                    {"gap_statement": "real gap"},
                ]
            }
        )
        assert [c.gap_statement for c in out.candidate_gaps] == ["real gap"]

    def test_schema_gap_type_enum_matches_taxonomy(self):
        schema = GapDetectionLLMOutput.json_schema_for_llm()
        item = schema["properties"]["candidate_gaps"]["items"]
        assert item["properties"]["gap_type"]["enum"] == [t.value for t in GapType]
        for field in ("supporting_chunk_ids", "initial_confidence", "population", "synonyms"):
            assert field in item["required"]


class TestM2LenientFieldsAndPerCandidateValidation:
    """M2 regression: lenient optional list fields, strict top level."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [("GNN", ["GNN"]), ("  ", []), (5, []), ({"a": 1}, []), (None, [])],
    )
    def test_list_field_coercion(self, value, expected):
        out = CandidateGapLLMOutput.model_validate(
            {"gap_statement": "g", "key_concepts": value, "secondary_gap_types": value}
        )
        assert out.key_concepts == expected
        assert out.secondary_gap_types == expected

    def test_non_string_optional_text_becomes_none(self):
        out = GapRepresentationLLMOutput.model_validate({"domain": ["NLP"], "outcome": 3})
        assert out.domain is None and out.outcome is None

    def test_invalid_items_dropped_and_described(self):
        out = GapDetectionLLMOutput.model_validate(
            {"candidate_gaps": [{"gap_statement": "ok"}, 42, {"gap_statement": None}]}
        )
        assert [c.gap_statement for c in out.candidate_gaps] == ["ok"]
        assert out.dropped_candidates == [
            "item 1: not an object (int)",
            "item 2: missing or blank gap_statement",
        ]

    def test_llm_cannot_inject_dropped_candidates(self):
        out = GapDetectionLLMOutput.model_validate(
            {"candidate_gaps": [{"gap_statement": "ok"}], "dropped_candidates": ["fake"]}
        )
        assert out.dropped_candidates == []

    @pytest.mark.parametrize("payload", [[], "text", 3])
    def test_top_level_must_be_object(self, payload):
        with pytest.raises(pydantic.ValidationError):
            GapDetectionLLMOutput.model_validate(payload)

    @pytest.mark.parametrize("value", ["nope", 5, {"gap_statement": "x"}])
    def test_candidate_gaps_must_be_list(self, value):
        with pytest.raises(pydantic.ValidationError):
            GapDetectionLLMOutput.model_validate({"candidate_gaps": value})


class TestContradictingChunkIds:
    @pytest.mark.parametrize(
        ("value", "expected"), [(None, []), ("c3", ["c3"]), ([" c3 ", "", 5], ["c3"]), (7, [])]
    )
    def test_lenient_coercion(self, value, expected):
        out = CandidateGapLLMOutput.model_validate(
            {"gap_statement": "g", "contradicting_chunk_ids": value}
        )
        assert out.contradicting_chunk_ids == expected

    def test_default_empty(self):
        assert CandidateGapLLMOutput.model_validate({"gap_statement": "g"}).contradicting_chunk_ids == []
