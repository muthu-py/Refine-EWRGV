"""
tests/unit/test_gap_detector.py
---------------------------------
Unit tests for GapDetector (candidate gap identification).

All LLM calls are mocked — no API keys or network access required.

Test coverage
-------------
* Valid LLM output → CandidateGaps with types, evidence, representation
* Invalid chunk IDs removed; ungrounded candidates dropped
* Empty input → [] without calling the LLM
* Provider errors / non-dict / malformed top-level output → GapDetectionError
* Malformed candidates dropped individually; lenient list fields (M2)
* Decorated chunk IDs resolved, unsupplied IDs rejected (L1)
* Gap-type mapping (exact, alias, unknown → OTHER)
* Candidate and chunk caps
* Prompt grounding (chunk IDs present; only selected chunks sent)
* Stub-corpus chunks as input
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from app.core.exceptions import GapDetectionError, ProviderError
from app.domain.enums import GapType
from app.domain.models.candidate_gap import CandidateGap, StructuredGapRepresentation
from app.domain.models.paper import DocumentChunk
from app.domain.models.query import ResearchQuery
from app.domain.models.query_understanding import QueryUnderstanding
from app.gap_detection.detector import GapDetector, map_gap_type, resolve_chunk_id
from app.stubs.stub_corpus import StubCorpusStore

# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _make_detector(
    llm_return_value: Any = None, raises: Exception | None = None, **kwargs: Any
) -> tuple[GapDetector, MagicMock]:
    mock_llm = MagicMock()
    if raises is not None:
        mock_llm.complete_structured.side_effect = raises
    else:
        mock_llm.complete_structured.return_value = llm_return_value
    return GapDetector(mock_llm, **kwargs), mock_llm


def _chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id="c1",
            paper_id="p1",
            section="Limitations",
            text="Our evaluation covers only English benchmarks.",
        ),
        DocumentChunk(
            chunk_id="c2",
            paper_id="p2",
            section="Future Work",
            text="Extending the approach to low-resource languages remains open.",
        ),
        DocumentChunk(
            chunk_id="c3",
            paper_id="p1",
            section="Results",
            text="The model achieves state-of-the-art accuracy on GLUE.",
        ),
    ]


def _gap(**overrides: Any) -> dict:
    gap = {
        "gap_statement": "Transformer models are rarely evaluated on low-resource languages.",
        "gap_type": "Population",
        "secondary_gap_types": ["Evaluation"],
        "supporting_chunk_ids": ["c1", "c2"],
        "initial_confidence": 0.75,
        "rationale": "Both passages state English-only evaluation.",
        "domain": "NLP",
        "population": "low-resource languages",
        "methodology": "Transformer",
        "outcome": "evaluation coverage",
        "temporal_scope": None,
        "geographic_scope": None,
        "key_concepts": ["Transformer", "low-resource languages"],
        "synonyms": ["under-resourced languages", "LRL"],
    }
    gap.update(overrides)
    return gap


QUERY = ResearchQuery(query="What are the gaps in multilingual Transformer evaluation?")


# ======================================================================
# Valid output
# ======================================================================


class TestGapDetectorValid:
    def test_returns_candidate_gaps(self):
        detector, _ = _make_detector({"candidate_gaps": [_gap()]})
        gaps = detector.detect(QUERY, _chunks())
        assert len(gaps) == 1
        assert isinstance(gaps[0], CandidateGap)

    def test_fields_mapped(self):
        detector, _ = _make_detector({"candidate_gaps": [_gap()]})
        gap = detector.detect(QUERY, _chunks())[0]
        assert gap.gap_statement.startswith("Transformer models")
        assert gap.primary_gap_type == GapType.POPULATION
        assert gap.secondary_gap_types == [GapType.EVALUATION]
        assert gap.relevant_passages == ["c1", "c2"]
        assert gap.source_papers == ["p1", "p2"]
        assert gap.initial_confidence == 0.75
        assert gap.rank is None
        assert gap.metadata["rationale"].startswith("Both passages")
        assert gap.metadata["query_id"] == QUERY.query_id
        assert "invalid_chunk_ids" not in gap.metadata

    def test_structured_representation_populated(self):
        detector, _ = _make_detector({"candidate_gaps": [_gap()]})
        rep = detector.detect(QUERY, _chunks())[0].structured_representation
        assert isinstance(rep, StructuredGapRepresentation)
        assert rep.gap_statement.startswith("Transformer models")
        assert rep.domain == "NLP"
        assert rep.population == "low-resource languages"
        assert rep.methodology == "Transformer"
        assert rep.key_concepts == ["Transformer", "low-resource languages"]
        assert rep.synonyms == ["under-resourced languages", "LRL"]

    def test_source_papers_deduplicated_in_citation_order(self):
        detector, _ = _make_detector(
            {"candidate_gaps": [_gap(supporting_chunk_ids=["c3", "c2", "c1", "c3"])]}
        )
        gap = detector.detect(QUERY, _chunks())[0]
        assert gap.relevant_passages == ["c3", "c2", "c1"]
        assert gap.source_papers == ["p1", "p2"]

    def test_multiple_gaps_preserve_llm_order(self):
        payload = {
            "candidate_gaps": [
                _gap(gap_statement="Gap A", supporting_chunk_ids=["c1"]),
                _gap(gap_statement="Gap B", supporting_chunk_ids=["c2"]),
            ]
        }
        detector, _ = _make_detector(payload)
        gaps = detector.detect(QUERY, _chunks())
        assert [g.gap_statement for g in gaps] == ["Gap A", "Gap B"]
        assert len({g.gap_id for g in gaps}) == 2

    def test_empty_candidate_list_is_valid(self):
        detector, _ = _make_detector({"candidate_gaps": []})
        assert detector.detect(QUERY, _chunks()) == []

    def test_confidence_clamped(self):
        detector, _ = _make_detector({"candidate_gaps": [_gap(initial_confidence=1.7)]})
        assert detector.detect(QUERY, _chunks())[0].initial_confidence == 1.0

    def test_works_on_stub_corpus_chunks(self):
        chunks = StubCorpusStore().get_all_chunks()
        detector, mock_llm = _make_detector(
            {"candidate_gaps": [_gap(supporting_chunk_ids=["chunk-003-limits"])]}
        )
        gap = detector.detect(QUERY, chunks)[0]
        assert gap.source_papers == ["paper-003"]
        prompt = mock_llm.complete_structured.call_args.args[0]
        assert "[chunk_id: chunk-003-limits]" in prompt
        assert "section: Limitations" in prompt


# ======================================================================
# Invalid chunk IDs (grounding)
# ======================================================================


class TestGapDetectorGrounding:
    def test_invalid_ids_removed_and_recorded(self):
        detector, _ = _make_detector(
            {"candidate_gaps": [_gap(supporting_chunk_ids=["c1", "ghost", "c9"])]}
        )
        gap = detector.detect(QUERY, _chunks())[0]
        assert gap.relevant_passages == ["c1"]
        assert gap.source_papers == ["p1"]
        assert gap.metadata["invalid_chunk_ids"] == ["ghost", "c9"]

    def test_candidate_with_only_invalid_ids_dropped(self):
        payload = {
            "candidate_gaps": [
                _gap(gap_statement="Ungrounded", supporting_chunk_ids=["ghost"]),
                _gap(gap_statement="Grounded", supporting_chunk_ids=["c2"]),
            ]
        }
        detector, _ = _make_detector(payload)
        gaps = detector.detect(QUERY, _chunks())
        assert [g.gap_statement for g in gaps] == ["Grounded"]

    def test_candidate_with_no_ids_dropped(self):
        detector, _ = _make_detector({"candidate_gaps": [_gap(supporting_chunk_ids=[])]})
        assert detector.detect(QUERY, _chunks()) == []

    def test_ids_of_chunks_not_sent_to_llm_are_invalid(self):
        """A chunk beyond max_chunks was never shown to the LLM, so it can't be cited."""
        detector, mock_llm = _make_detector(
            {"candidate_gaps": [_gap(supporting_chunk_ids=["c3"])]}, max_chunks=2
        )
        assert detector.detect(QUERY, _chunks()) == []
        assert "c3" not in mock_llm.complete_structured.call_args.args[0]


# ======================================================================
# Empty input
# ======================================================================


class TestGapDetectorEmptyInput:
    def test_empty_chunks_returns_empty_without_llm_call(self):
        detector, mock_llm = _make_detector({"candidate_gaps": [_gap()]})
        assert detector.detect(QUERY, []) == []
        mock_llm.complete_structured.assert_not_called()

    def test_blank_text_chunks_treated_as_empty(self):
        detector, mock_llm = _make_detector({"candidate_gaps": [_gap()]})
        blank = [DocumentChunk(chunk_id="b", paper_id="p", text="   ")]
        assert detector.detect(QUERY, blank) == []
        mock_llm.complete_structured.assert_not_called()

    def test_duplicate_input_chunks_sent_once(self):
        detector, mock_llm = _make_detector({"candidate_gaps": []})
        chunks = _chunks()
        detector.detect(QUERY, chunks + chunks)
        prompt = mock_llm.complete_structured.call_args.args[0]
        assert prompt.count("[chunk_id: c1]") == 1


# ======================================================================
# Errors
# ======================================================================


class TestGapDetectorErrors:
    def test_provider_error_wrapped(self):
        detector, _ = _make_detector(raises=ProviderError("timeout"))
        with pytest.raises(GapDetectionError) as exc_info:
            detector.detect(QUERY, _chunks())
        assert isinstance(exc_info.value.__cause__, ProviderError)
        assert "timeout" in str(exc_info.value)  # M3: cause not swallowed

    def test_unexpected_exception_wrapped(self):
        detector, _ = _make_detector(raises=RuntimeError("boom"))
        with pytest.raises(GapDetectionError):
            detector.detect(QUERY, _chunks())

    def test_non_dict_response(self):
        detector, _ = _make_detector(["not", "a", "dict"])
        with pytest.raises(GapDetectionError):
            detector.detect(QUERY, _chunks())

    def test_malformed_candidate_list(self):
        detector, _ = _make_detector({"candidate_gaps": "nope"})
        with pytest.raises(GapDetectionError):
            detector.detect(QUERY, _chunks())

    def test_malformed_candidate_item(self):
        detector, _ = _make_detector({"candidate_gaps": ["just a string"]})
        with pytest.raises(GapDetectionError):
            detector.detect(QUERY, _chunks())

    def test_invalid_constructor_arguments(self):
        with pytest.raises(ValueError):
            GapDetector(MagicMock(), max_candidates=0)


# ======================================================================
# Gap types, caps, prompt
# ======================================================================


class TestGapDetectorMappingAndCaps:
    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("Methodological", GapType.METHODOLOGICAL),
            ("methodological", GapType.METHODOLOGICAL),
            ("CONTRADICTORY_EVIDENCE", GapType.CONTRADICTORY_EVIDENCE),
            ("Contradictory Evidence", GapType.CONTRADICTORY_EVIDENCE),
            ("Other / Unknown", GapType.OTHER),
            ("dataset", GapType.DATASET),
            ("astrology", None),
            (None, None),
        ],
    )
    def test_map_gap_type(self, label, expected):
        assert map_gap_type(label) == expected

    def test_unknown_type_falls_back_to_other(self):
        detector, _ = _make_detector({"candidate_gaps": [_gap(gap_type="Societal")]})
        gap = detector.detect(QUERY, _chunks())[0]
        assert gap.primary_gap_type == GapType.OTHER
        assert gap.metadata["raw_gap_type"] == "Societal"

    def test_secondary_types_exclude_primary_and_unknown(self):
        detector, _ = _make_detector(
            {
                "candidate_gaps": [
                    _gap(secondary_gap_types=["Population", "Dataset", "bogus", "Dataset"])
                ]
            }
        )
        gap = detector.detect(QUERY, _chunks())[0]
        assert gap.secondary_gap_types == [GapType.DATASET]

    def test_max_candidates_cap(self):
        payload = {
            "candidate_gaps": [_gap(gap_statement=f"Gap {i}") for i in range(10)]
        }
        detector, mock_llm = _make_detector(payload, max_candidates=3)
        assert len(detector.detect(QUERY, _chunks())) == 3
        assert "at most 3 candidate" in mock_llm.complete_structured.call_args.args[0]

    def test_prompt_includes_query_understanding(self):
        query = QUERY.model_copy(
            update={"understanding": QueryUnderstanding(domain="NLP", methods=["BERT"])}
        )
        detector, mock_llm = _make_detector({"candidate_gaps": []})
        detector.detect(query, _chunks())
        prompt = mock_llm.complete_structured.call_args.args[0]
        assert QUERY.query in prompt
        assert "- Domain: NLP" in prompt
        assert "- Methods: BERT" in prompt

    def test_long_chunk_truncated_in_prompt(self):
        long_chunk = DocumentChunk(chunk_id="long", paper_id="p", text="x" * 500)
        detector, mock_llm = _make_detector({"candidate_gaps": []}, max_chunk_chars=100)
        detector.detect(QUERY, [long_chunk])
        prompt = mock_llm.complete_structured.call_args.args[0]
        assert "x" * 101 not in prompt

    def test_schema_passed_to_provider(self):
        detector, mock_llm = _make_detector({"candidate_gaps": []})
        detector.detect(QUERY, _chunks())
        schema = mock_llm.complete_structured.call_args.args[1]
        assert "candidate_gaps" in schema["properties"]


# ======================================================================
# M2 regression — lenient list fields, per-candidate validation
# ======================================================================


class TestGapDetectorMalformedCandidates:
    @pytest.mark.parametrize(
        ("field", "value", "expected"),
        [
            ("secondary_gap_types", "Dataset", [GapType.DATASET]),
            ("secondary_gap_types", {"oops": 1}, []),
            ("secondary_gap_types", 7, []),
        ],
    )
    def test_malformed_secondary_types_handled(self, field, value, expected):
        detector, _ = _make_detector({"candidate_gaps": [_gap(**{field: value})]})
        gap = detector.detect(QUERY, _chunks())[0]
        assert gap.secondary_gap_types == expected

    @pytest.mark.parametrize(
        ("field", "value", "expected"),
        [
            ("key_concepts", "GNN", ["GNN"]),
            ("synonyms", "Controller Area Network", ["Controller Area Network"]),
            ("key_concepts", 42, []),
        ],
    )
    def test_malformed_representation_lists_handled(self, field, value, expected):
        detector, _ = _make_detector({"candidate_gaps": [_gap(**{field: value})]})
        rep = detector.detect(QUERY, _chunks())[0].structured_representation
        assert getattr(rep, field) == expected

    def test_bare_string_chunk_id_is_grounded(self):
        detector, _ = _make_detector({"candidate_gaps": [_gap(supporting_chunk_ids="c2")]})
        gap = detector.detect(QUERY, _chunks())[0]
        assert gap.relevant_passages == ["c2"]

    def test_non_string_optional_text_becomes_none(self):
        detector, _ = _make_detector(
            {"candidate_gaps": [_gap(domain=["NLP"], gap_type=["Dataset"])]}
        )
        gap = detector.detect(QUERY, _chunks())[0]
        assert gap.structured_representation.domain is None
        assert gap.primary_gap_type == GapType.OTHER

    def test_one_malformed_candidate_does_not_discard_valid_ones(self, caplog):
        payload = {
            "candidate_gaps": [
                _gap(gap_statement="Valid A", supporting_chunk_ids=["c1"]),
                "just a string",
                {"gap_statement": "   "},
                {"no_statement": True},
                _gap(gap_statement="Valid B", supporting_chunk_ids=["c2"]),
            ]
        }
        detector, _ = _make_detector(payload)
        with caplog.at_level("WARNING", logger="app.gap_detection.detector"):
            gaps = detector.detect(QUERY, _chunks())
        assert [g.gap_statement for g in gaps] == ["Valid A", "Valid B"]
        record = next(r for r in caplog.records if "malformed" in r.getMessage())
        assert len(record.dropped) == 3
        assert record.dropped[0].startswith("item 1: not an object")

    def test_all_candidates_malformed_raises(self):
        detector, _ = _make_detector({"candidate_gaps": ["junk", {"gap_statement": ""}]})
        with pytest.raises(GapDetectionError) as exc_info:
            detector.detect(QUERY, _chunks())
        assert "item 0" in exc_info.value.detail

    @pytest.mark.parametrize("payload", [{"candidate_gaps": "nope"}, {"candidate_gaps": 5}])
    def test_top_level_still_strict(self, payload):
        detector, _ = _make_detector(payload)
        with pytest.raises(GapDetectionError):
            detector.detect(QUERY, _chunks())


# ======================================================================
# L1 regression — decorated chunk IDs
# ======================================================================


class TestChunkIdResolution:
    @pytest.mark.parametrize(
        "cited",
        ["c1", " c1 ", "[c1]", "chunk_id: c1", "[chunk_id: c1]", "chunk-id=c1",
         "Chunk ID: c1", "`c1`", "'c1'", '"c1"', "[`chunk_id: c1`]", "id: c1"],
    )
    def test_harmless_decoration_accepted(self, cited):
        assert resolve_chunk_id(cited, {"c1", "c2"}) == "c1"

    @pytest.mark.parametrize(
        "cited",
        ["C1", "c1x", "c", "chunk_id: ghost", "[c3]", "", "[]", "chunk_id:", "c1, c2"],
    )
    def test_unsupplied_or_altered_ids_rejected(self, cited):
        assert resolve_chunk_id(cited, {"c1", "c2"}) is None

    def test_exact_match_wins_for_ids_containing_decoration(self):
        assert resolve_chunk_id("[weird-id]", {"[weird-id]", "weird-id"}) == "[weird-id]"

    def test_detector_resolves_decorated_ids_and_records_raw_invalid(self):
        detector, _ = _make_detector(
            {"candidate_gaps": [_gap(supporting_chunk_ids=["[chunk_id: c1]", "chunk_id: c2", "C3"])]}
        )
        gap = detector.detect(QUERY, _chunks())[0]
        assert gap.relevant_passages == ["c1", "c2"]
        assert gap.source_papers == ["p1", "p2"]
        assert gap.metadata["invalid_chunk_ids"] == ["C3"]

    def test_decorated_id_of_unsent_chunk_still_rejected(self):
        """Decoration stripping must not bypass the max_chunks grounding limit."""
        detector, _ = _make_detector(
            {"candidate_gaps": [_gap(supporting_chunk_ids=["[chunk_id: c3]"])]}, max_chunks=2
        )
        assert detector.detect(QUERY, _chunks()) == []


# ======================================================================
# Contradicting (counter-evidence) citations
# ======================================================================


class TestContradictingCitations:
    def test_contradicting_ids_resolved_and_stored(self):
        detector, _ = _make_detector({"candidate_gaps": [
            _gap(supporting_chunk_ids=["c1"], contradicting_chunk_ids=["[chunk_id: c3]"])
        ]})
        gap = detector.detect(QUERY, _chunks())[0]
        assert gap.relevant_passages == ["c1"]
        assert gap.contradicting_passages == ["c3"]
        assert gap.source_papers == ["p1"]  # counter evidence is not a source of the gap

    def test_missing_field_defaults_to_empty(self):
        detector, _ = _make_detector({"candidate_gaps": [_gap()]})
        assert detector.detect(QUERY, _chunks())[0].contradicting_passages == []

    def test_unsupplied_contradicting_ids_recorded(self):
        detector, _ = _make_detector({"candidate_gaps": [
            _gap(contradicting_chunk_ids=["ghost", "c3"])
        ]})
        gap = detector.detect(QUERY, _chunks())[0]
        assert gap.contradicting_passages == ["c3"]
        assert gap.metadata["invalid_contradicting_chunk_ids"] == ["ghost"]

    def test_passage_in_both_lists_kept_only_as_counter_evidence(self):
        detector, _ = _make_detector({"candidate_gaps": [
            _gap(supporting_chunk_ids=["c1", "c2"], contradicting_chunk_ids=["c2"])
        ]})
        gap = detector.detect(QUERY, _chunks())[0]
        assert gap.relevant_passages == ["c1"]
        assert gap.contradicting_passages == ["c2"]
        assert gap.metadata["ambiguous_chunk_ids"] == ["c2"]

    def test_candidate_whose_only_support_is_ambiguous_is_dropped(self):
        detector, _ = _make_detector({"candidate_gaps": [
            _gap(supporting_chunk_ids=["c1"], contradicting_chunk_ids=["c1"])
        ]})
        assert detector.detect(QUERY, _chunks()) == []

    def test_prompt_and_schema_request_counter_evidence(self):
        detector, mock_llm = _make_detector({"candidate_gaps": []})
        detector.detect(QUERY, _chunks())
        prompt, schema = mock_llm.complete_structured.call_args.args
        assert "### contradicting_chunk_ids" in prompt
        item = schema["properties"]["candidate_gaps"]["items"]
        assert "contradicting_chunk_ids" in item["properties"]
