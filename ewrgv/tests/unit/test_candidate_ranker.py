"""
tests/unit/test_candidate_ranker.py
-------------------------------------
Unit tests for CandidateRanker (deterministic heuristic ranking).

Test coverage
-------------
* Base score: confidence and evidence breadth (distinct papers / passages)
* Near-duplicate removal (exact and threshold-based)
* Gap-type diversity re-ranking
* top_k, empty input, determinism, immutability of inputs
* Configuration validation
"""

from __future__ import annotations

import pytest

from app.domain.enums import GapType
from app.domain.models.candidate_gap import CandidateGap
from app.gap_detection.ranking import CandidateRanker


def _gap(
    statement: str,
    confidence: float = 0.5,
    gap_type: GapType = GapType.METHODOLOGICAL,
    papers: list[str] | None = None,
    passages: list[str] | None = None,
) -> CandidateGap:
    return CandidateGap(
        gap_statement=statement,
        primary_gap_type=gap_type,
        initial_confidence=confidence,
        source_papers=papers if papers is not None else ["p1"],
        relevant_passages=passages if passages is not None else ["c1"],
    )


class TestBaseScore:
    def test_confidence_orders_candidates(self):
        gaps = [
            _gap("alpha topic gap", 0.2),
            _gap("beta topic gap", 0.9),
            _gap("gamma topic gap", 0.5),
        ]
        ranked = CandidateRanker(diversity_penalty=0).rank(gaps, top_k=3)
        assert [g.initial_confidence for g in ranked] == [0.9, 0.5, 0.2]
        assert [g.rank for g in ranked] == [1, 2, 3]

    def test_evidence_breadth_breaks_equal_confidence(self):
        narrow = _gap("narrow evidence gap", 0.5, papers=["p1"], passages=["c1"])
        broad = _gap(
            "broad evidence gap", 0.5,
            papers=["p1", "p2", "p3"], passages=["c1", "c2", "c3", "c4", "c5"],
        )
        ranked = CandidateRanker().rank([narrow, broad], top_k=2)
        assert ranked[0].gap_statement == "broad evidence gap"
        assert ranked[0].metadata["ranking"]["distinct_papers"] == 3
        assert ranked[0].metadata["ranking"]["distinct_passages"] == 5

    def test_duplicate_evidence_ids_counted_once(self):
        ranker = CandidateRanker()
        a = _gap("a gap", papers=["p1", "p1"], passages=["c1", "c1"])
        b = _gap("b gap", papers=["p1"], passages=["c1"])
        assert ranker.base_score(a) == pytest.approx(ranker.base_score(b))

    def test_base_score_bounds(self):
        ranker = CandidateRanker()
        top = _gap("x", 1.0, papers=[f"p{i}" for i in range(10)],
                   passages=[f"c{i}" for i in range(10)])
        assert ranker.base_score(top) == pytest.approx(1.0)
        assert ranker.base_score(_gap("y", 0.0, papers=[], passages=[])) == 0.0

    def test_weights_are_configurable(self):
        confident = _gap("confident gap", 0.9, papers=["p1"], passages=["c1"])
        broad = _gap("broad gap", 0.1, papers=["p1", "p2", "p3"],
                     passages=["c1", "c2", "c3", "c4", "c5"])
        default = CandidateRanker().rank([confident, broad], top_k=1)
        evidence_only = CandidateRanker(confidence_weight=0).rank([confident, broad], top_k=1)
        assert default[0].gap_statement == "confident gap"
        assert evidence_only[0].gap_statement == "broad gap"


class TestDuplicates:
    def test_exact_duplicates_removed_keeping_higher_score(self):
        low = _gap("Transformers are rarely evaluated on low-resource languages.", 0.3)
        high = _gap("transformers are rarely evaluated on low-resource languages!", 0.8)
        ranked = CandidateRanker().rank([low, high], top_k=5)
        assert len(ranked) == 1
        assert ranked[0].gap_id == high.gap_id
        assert ranked[0].metadata["ranking"]["merged_duplicates"] == [low.gap_id]

    def test_near_duplicate_threshold_configurable(self):
        a = _gap("Transformer models are rarely evaluated on low-resource languages", 0.8)
        b = _gap("Transformer models are seldom evaluated on low-resource languages", 0.6)
        # Token Jaccard = 6/8 = 0.75
        assert len(CandidateRanker(duplicate_threshold=0.8).rank([a, b], top_k=5)) == 2
        ranked = CandidateRanker(duplicate_threshold=0.7).rank([a, b], top_k=5)
        assert [g.gap_id for g in ranked] == [a.gap_id]

    def test_distinct_gaps_not_merged(self):
        a = _gap("No dataset exists for Tamil speech emotion recognition", 0.5)
        b = _gap("GNN intrusion detection is untested on real vehicles", 0.5)
        assert len(CandidateRanker().rank([a, b], top_k=5)) == 2


class TestDiversity:
    def _gaps(self) -> list[CandidateGap]:
        return [
            _gap("method gap alpha", 0.9, GapType.METHODOLOGICAL),
            _gap("method gap beta", 0.85, GapType.METHODOLOGICAL),
            _gap("dataset gap gamma", 0.8, GapType.DATASET),
        ]

    def test_diversity_promotes_other_gap_type(self):
        ranked = CandidateRanker(diversity_penalty=0.1).rank(self._gaps(), top_k=3)
        assert [g.primary_gap_type for g in ranked] == [
            GapType.METHODOLOGICAL, GapType.DATASET, GapType.METHODOLOGICAL,
        ]
        assert ranked[2].metadata["ranking"]["diversity_penalty"] == pytest.approx(0.1)

    def test_zero_penalty_is_pure_score_order(self):
        ranked = CandidateRanker(diversity_penalty=0).rank(self._gaps(), top_k=3)
        assert [g.initial_confidence for g in ranked] == [0.9, 0.85, 0.8]

    def test_diversity_respects_top_k(self):
        ranked = CandidateRanker(diversity_penalty=0.1).rank(self._gaps(), top_k=2)
        assert {g.primary_gap_type for g in ranked} == {GapType.METHODOLOGICAL, GapType.DATASET}


class TestRankContract:
    def test_empty_input(self):
        assert CandidateRanker().rank([], top_k=5) == []

    def test_non_positive_top_k(self):
        assert CandidateRanker().rank([_gap("g")], top_k=0) == []

    def test_top_k_larger_than_input(self):
        assert len(CandidateRanker().rank([_gap("a"), _gap("b c")], top_k=10)) == 2

    def test_inputs_not_mutated(self):
        gaps = [_gap("first gap", 0.4), _gap("second gap", 0.6)]
        ranked = CandidateRanker().rank(gaps, top_k=2)
        assert all(g.rank is None and "ranking" not in g.metadata for g in gaps)
        assert {g.gap_id for g in ranked} == {g.gap_id for g in gaps}

    def test_existing_metadata_preserved(self):
        gap = _gap("g")
        gap.metadata["rationale"] = "keep me"
        ranked = CandidateRanker().rank([gap], top_k=1)
        assert ranked[0].metadata["rationale"] == "keep me"

    def test_deterministic_regardless_of_input_order(self):
        gaps = [_gap(f"distinct statement {w}", 0.5) for w in ("one", "two", "three", "four")]
        first = [g.gap_id for g in CandidateRanker().rank(gaps, top_k=4)]
        second = [g.gap_id for g in CandidateRanker().rank(list(reversed(gaps)), top_k=4)]
        assert first == second

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"confidence_weight": -1},
            {"confidence_weight": 0, "paper_weight": 0, "passage_weight": 0},
            {"paper_saturation": 0},
            {"duplicate_threshold": 0},
            {"duplicate_threshold": 1.5},
            {"diversity_penalty": -0.1},
        ],
    )
    def test_invalid_configuration(self, kwargs):
        with pytest.raises(ValueError):
            CandidateRanker(**kwargs)


class TestUnicodeDuplicates:
    """M1 regression: non-ASCII statements must not all collapse as duplicates."""

    def test_distinct_non_latin_statements_kept(self):
        gaps = [
            _gap("தமிழ் பேச்சு உணர்வு தரவுத்தொகுப்பு இல்லை", 0.9),
            _gap("低资源语言缺乏评估基准", 0.8),
            _gap("Ελλείψη δεδομένων για ελληνικά", 0.7),
            _gap("हिंदी भाषण पहचान के लिए डेटासेट नहीं", 0.6),
        ]
        assert len(CandidateRanker().rank(gaps, top_k=10)) == 4

    def test_distinct_tamil_statements_sharing_words_kept(self):
        a = _gap("தமிழ் பேச்சு தரவுத்தொகுப்பு இல்லை", 0.9)
        b = _gap("தமிழ் உரை மதிப்பீடு அளவுகோல் இல்லை", 0.8)
        assert len(CandidateRanker().rank([a, b], top_k=5)) == 2

    def test_identical_non_latin_statements_merged(self):
        a = _gap("低资源语言缺乏评估基准", 0.9)
        b = _gap("低资源语言缺乏评估基准。", 0.5)
        ranked = CandidateRanker().rank([a, b], top_k=5)
        assert [g.gap_id for g in ranked] == [a.gap_id]
        assert ranked[0].metadata["ranking"]["merged_duplicates"] == [b.gap_id]

    def test_accented_words_kept_whole(self):
        a = _gap("Évaluation des modèles de langue", 0.9)
        b = _gap("Évaluation des données de langue", 0.8)
        assert len(CandidateRanker().rank([a, b], top_k=5)) == 2

    def test_composed_and_decomposed_accents_are_duplicates(self):
        composed = _gap("Évaluation des modèles", 0.9)
        decomposed = _gap("Évaluation des modèles", 0.5)
        assert len(CandidateRanker().rank([composed, decomposed], top_k=5)) == 1

    def test_tokenless_statements_not_merged_unless_identical(self):
        gaps = [_gap("???", 0.9), _gap("!!!", 0.8), _gap("—", 0.7)]
        assert len(CandidateRanker().rank(gaps, top_k=5)) == 3

    def test_stopword_only_statements_not_merged(self):
        a = _gap("the of and", 0.9)
        b = _gap("it is not", 0.8)
        assert len(CandidateRanker().rank([a, b], top_k=5)) == 2


class TestNaNConfidence:
    """L2 regression: NaN confidence must not distort ranking."""

    def test_nan_scored_as_zero(self):
        ranker = CandidateRanker()
        nan_gap = _gap("nan gap", float("nan"))
        zero_gap = _gap("zero gap", 0.0)
        assert ranker.base_score(nan_gap) == pytest.approx(ranker.base_score(zero_gap))

    def test_nan_does_not_rank_first(self):
        gaps = [_gap("aaa nan gap", float("nan")), _gap("bbb high", 0.9), _gap("ccc low", 0.1)]
        ranked = CandidateRanker(diversity_penalty=0).rank(gaps, top_k=3)
        assert [g.gap_statement for g in ranked] == ["bbb high", "ccc low", "aaa nan gap"]
        assert all(g.metadata["ranking"]["base_score"] == g.metadata["ranking"]["base_score"]
                   for g in ranked)  # no NaN in output

    @pytest.mark.parametrize(("value", "expected"), [(float("inf"), 1.0), (float("-inf"), 0.0)])
    def test_infinite_confidence_clamped(self, value, expected):
        ranker = CandidateRanker(paper_weight=0, passage_weight=0)
        assert ranker.base_score(_gap("g", value)) == pytest.approx(expected)
