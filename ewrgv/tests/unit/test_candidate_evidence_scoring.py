"""
tests/unit/test_candidate_evidence_scoring.py
-----------------------------------------------
Unit tests for CandidateEvidenceScorer (pre-validation evidence scoring).

Deterministic and offline.  Expected values are computed by hand from the
documented formula (see app/gap_detection/evidence_scoring.py).

Fixture (gap terms = {alpha, beta, gamma, delta}; default weights
relevance .35, quality .20, recency .15, independence .30; reference year =
latest evidence year = 2020; horizon 10 years):

    c1  supporting  Limitations  "alpha beta gamma delta"  2020  p1
        relevance 1.0, quality 1.0, recency 1.0 → strength 1.0
    c2  supporting  Abstract     "alpha beta"              2010  p2
        relevance 0.5, quality 0.5, recency 0.0 → strength (.175+.1)/.7 = .392857
    c3  counter     Results      "alpha beta gamma"        2020  p3
        relevance .75, recency 1.0          → strength (.2625+.15)/.5 = .825

    support features: relevance .75, quality .75, recency .5, independence 2/3
    support_score = .35·.75 + .2·.75 + .15·.5 + .3·(2/3) = .6875
"""

from __future__ import annotations

import pytest

from app.domain.enums import CandidateEvidenceStatus, EvidenceType
from app.domain.models.candidate_gap import CandidateGap, StructuredGapRepresentation
from app.domain.models.paper import Paper
from app.domain.models.paper_chunk import PaperChunk
from app.gap_detection.evidence_scoring import DEFAULT_WEIGHTS, CandidateEvidenceScorer
from app.gap_detection.ranking import CandidateRanker
from tests.fixtures.sample_corpus import make_sample_corpus
from tests.fixtures.sample_data import make_chunk

STATEMENT = "alpha beta gamma delta"

# Publication year and title live on the Paper, not the chunk; tests register
# each chunk's paper here and pass LOOKUP as the scorer's paper_lookup.
PAPERS: dict[str, Paper] = {}
LOOKUP = PAPERS.get


def _chunk(cid: str, pid: str, section: str, text: str, year=None, title=None) -> PaperChunk:
    PAPERS[pid] = Paper(paper_id=pid, title=title or f"Paper {pid}", publication_year=year)
    return make_chunk(chunk_id=cid, paper_id=pid, section=section, text=text)


def _chunks() -> dict[str, PaperChunk]:
    chunks = [
        _chunk("c1", "p1", "Limitations", "alpha beta gamma delta", year=2020, title="Paper One"),
        _chunk("c2", "p2", "Abstract", "alpha beta", year=2010),
        _chunk("c3", "p3", "Results", "alpha beta gamma", year=2020),
        _chunk("c4", "p4", "Introduction", "alpha"),
    ]
    return {c.chunk_id: c for c in chunks}


def _gap(supporting=("c1", "c2"), contradicting=(), statement=STATEMENT, **kw) -> CandidateGap:
    return CandidateGap(
        gap_statement=statement,
        relevant_passages=list(supporting),
        contradicting_passages=list(contradicting),
        **kw,
    )


def _score(gap: CandidateGap, **scorer_kwargs):
    return CandidateEvidenceScorer(**scorer_kwargs).score(gap, _chunks(), LOOKUP)


# ======================================================================
# Domain models
# ======================================================================


class TestDomainModels:
    def test_candidate_gap_new_fields_default_empty(self):
        gap = CandidateGap(gap_statement="g")
        assert gap.contradicting_passages == []
        assert gap.evidence_assessment is None

    def test_assessment_cannot_claim_validation(self):
        import pydantic

        from app.domain.models.candidate_evidence import CandidateEvidenceAssessment

        with pytest.raises(pydantic.ValidationError):
            CandidateEvidenceAssessment(
                score=1.0, status=CandidateEvidenceStatus.SUPPORTED,
                support_score=1.0, contradiction_ratio=0.0, is_validated=True,
            )


# ======================================================================
# Formula
# ======================================================================


class TestScoringFormula:
    def test_supported_candidate_exact_score(self):
        a = _score(_gap())
        assert a.status == CandidateEvidenceStatus.SUPPORTED
        assert a.features == pytest.approx(
            {"relevance": 0.75, "quality": 0.75, "recency": 0.5, "independence": 2 / 3},
            abs=1e-6,
        )
        assert a.support_score == pytest.approx(0.6875, abs=1e-6)
        assert a.contradiction_ratio == 0.0
        assert a.score == pytest.approx(0.6875, abs=1e-6)
        assert a.weights == pytest.approx(DEFAULT_WEIGHTS)

    def test_item_features(self):
        a = _score(_gap())
        c1, c2 = a.supporting
        assert (c1.relevance, c1.quality, c1.recency, c1.strength) == (1.0, 1.0, 1.0, 1.0)
        assert (c2.relevance, c2.quality, c2.recency) == (0.5, 0.5, 0.0)
        assert c2.strength == pytest.approx(0.392857, abs=1e-6)
        assert c1.stance == EvidenceType.SUPPORTING
        assert c1.title == "Paper One" and c1.publication_year == 2020

    def test_references_point_to_retrieved_passages(self):
        a = _score(_gap())
        assert [(i.chunk_id, i.paper_id) for i in a.supporting] == [("c1", "p1"), ("c2", "p2")]

    def test_relevance_uses_key_concepts(self):
        gap = _gap(supporting=("c4",), statement="alpha",
                   structured_representation=StructuredGapRepresentation(
                       gap_statement="alpha", key_concepts=["zeta"]))
        # gap terms {alpha, zeta}; c4 contains only "alpha"
        assert _score(gap).supporting[0].relevance == 0.5

    def test_independence_saturates(self):
        a = _score(_gap(supporting=("c1", "c2", "c4")), independence_saturation=2)
        assert a.features["independence"] == 1.0

    def test_scores_bounded_and_deterministic(self):
        a1, a2 = _score(_gap(contradicting=("c3",))), _score(_gap(contradicting=("c3",)))
        assert a1 == a2
        assert 0.0 <= a1.score <= a1.support_score <= 1.0

    def test_never_marked_validated(self):
        assert _score(_gap()).is_validated is False


# ======================================================================
# Contradictory evidence
# ======================================================================


class TestContradictoryEvidence:
    def test_contested_penalty_exact(self):
        a = _score(_gap(contradicting=("c3",)))
        # ratio = .825 / (1.0 + .392857 + .825)
        assert a.contradiction_ratio == pytest.approx(0.371981, abs=1e-6)
        assert a.score == pytest.approx(0.6875 * (1 - 0.371981), abs=1e-6)
        assert a.status == CandidateEvidenceStatus.CONTESTED
        assert a.contradicting[0].stance == EvidenceType.COUNTER
        assert a.contradicting[0].quality is None
        assert a.contradicting[0].strength == pytest.approx(0.825, abs=1e-6)

    def test_contradicted_when_counter_outweighs_support(self):
        a = _score(_gap(supporting=("c2",), contradicting=("c3",)))
        assert a.contradiction_ratio == pytest.approx(0.825 / (0.392857 + 0.825), abs=1e-5)
        assert a.status == CandidateEvidenceStatus.CONTRADICTED
        assert a.score < a.support_score / 2

    def test_contradiction_penalty_configurable(self):
        half = _score(_gap(contradicting=("c3",)), contradiction_penalty=0.5)
        none = _score(_gap(contradicting=("c3",)), contradiction_penalty=0.0)
        assert half.score == pytest.approx(0.6875 * (1 - 0.5 * 0.371981), abs=1e-6)
        assert none.score == pytest.approx(0.6875, abs=1e-6)
        assert none.status == CandidateEvidenceStatus.CONTESTED  # still reported

    def test_passage_in_both_lists_is_not_support(self):
        a = _score(_gap(supporting=("c1",), contradicting=("c1",)))
        assert a.status == CandidateEvidenceStatus.INSUFFICIENT_EVIDENCE
        assert a.supporting == [] and [i.chunk_id for i in a.contradicting] == ["c1"]

    def test_explanation_mentions_contradiction(self):
        text = _score(_gap(contradicting=("c3",))).explanation
        assert "1 contradicting passage(s)" in text and "not a validation" in text


# ======================================================================
# Missing evidence
# ======================================================================


class TestMissingEvidence:
    def test_no_supporting_passages_scores_zero(self):
        a = _score(_gap(supporting=()))
        assert a.score == 0.0
        assert a.status == CandidateEvidenceStatus.INSUFFICIENT_EVIDENCE
        assert a.features == dict.fromkeys(("relevance", "quality", "recency", "independence"))
        assert "Missing evidence is not evidence that the gap exists" in a.explanation

    def test_cited_but_unavailable_passages_recorded(self):
        a = _score(_gap(supporting=("ghost", "c1")))
        assert a.missing_chunk_ids == ["ghost"]
        assert [i.chunk_id for i in a.supporting] == ["c1"]
        assert "1 cited passage(s) were not available" in a.explanation

    def test_only_unavailable_passages_is_insufficient(self):
        a = _score(_gap(supporting=("ghost",)))
        assert a.status == CandidateEvidenceStatus.INSUFFICIENT_EVIDENCE
        assert a.score == 0.0 and a.missing_chunk_ids == ["ghost"]

    def test_only_counter_evidence(self):
        a = _score(_gap(supporting=(), contradicting=("c3",)))
        assert a.status == CandidateEvidenceStatus.INSUFFICIENT_EVIDENCE
        assert a.score == 0.0 and a.contradiction_ratio == 1.0
        assert "may already be addressed" in a.explanation

    def test_absence_of_counter_evidence_does_not_raise_score(self):
        assert _score(_gap()).score == _score(_gap()).support_score
        assert "does not confirm the gap" in _score(_gap()).explanation

    def test_missing_years_exclude_recency_and_renormalise(self):
        a = _score(_gap(supporting=("c4",)))  # c4 has no year
        assert a.features["recency"] is None
        assert "recency" not in a.weights
        assert sum(a.weights.values()) == pytest.approx(1.0)
        assert a.supporting[0].recency is None
        assert "recency unavailable" in a.explanation


# ======================================================================
# Weight configuration
# ======================================================================


class TestWeightConfiguration:
    def test_single_feature_weight(self):
        a = _score(_gap(), weights={"relevance": 1.0})
        assert a.score == pytest.approx(0.75, abs=1e-6)
        assert a.weights == {"relevance": 1.0}

    def test_weights_are_normalised(self):
        doubled = {f: 2 * w for f, w in DEFAULT_WEIGHTS.items()}
        assert _score(_gap(), weights=doubled).score == pytest.approx(0.6875, abs=1e-6)

    def test_weights_change_ranking_signal(self):
        # relevance .75 and recency .5, equally weighted → .625
        recency_heavy = _score(_gap(), weights={"relevance": 1.0, "recency": 1.0})
        assert recency_heavy.score == pytest.approx(0.625, abs=1e-6)
        assert recency_heavy.weights == {"relevance": 0.5, "recency": 0.5}

    def test_reference_year_configurable(self):
        a = _score(_gap(), reference_year=2025)
        assert [i.recency for i in a.supporting] == pytest.approx([0.5, 0.0])

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"weights": {"citations": 1.0}},
            {"weights": {"relevance": -0.1}},
            {"weights": {"relevance": 0.0}},
            {"weights": {"relevance": float("nan")}},
            {"independence_saturation": 0},
            {"recency_horizon_years": 0},
            {"contradiction_penalty": 1.5},
            {"explicit_section_quality": 2.0},
        ],
    )
    def test_invalid_configuration(self, kwargs):
        with pytest.raises(ValueError):
            CandidateEvidenceScorer(**kwargs)


# ======================================================================
# score_all, sample corpus, ranking
# ======================================================================


class TestScoreAllAndRanking:
    def test_score_all_returns_copies(self):
        gap = _gap()
        [scored] = CandidateEvidenceScorer().score_all([gap], list(_chunks().values()), LOOKUP)
        assert gap.evidence_assessment is None
        assert scored.evidence_assessment is not None
        assert scored.gap_id == gap.gap_id

    def test_sample_corpus_with_paper_lookup(self):
        """Chunks carry no year; the paper lookup supplies it."""
        store = make_sample_corpus()
        gap = _gap(supporting=("chunk-003-limits",),
                   statement="GPT-3 has limitations in structured prediction")
        [scored] = CandidateEvidenceScorer().score_all(
            [gap], store.get_all_chunks(), store.get_paper_by_id
        )
        item = scored.evidence_assessment.supporting[0]
        assert item.quality == 1.0  # Limitations section
        assert item.publication_year == 2020
        assert item.citation_count is None  # not invented

    def test_evidence_score_changes_ranking(self):
        # Equal heuristic scores; the WEAKER candidate sorts first alphabetically,
        # so only the evidence score can put the stronger one on top.
        strong = _gap(statement="zz strong alpha beta gamma delta", initial_confidence=0.5)
        weak = _gap(statement="aa weak alpha beta gamma delta", initial_confidence=0.5,
                    contradicting=("c3",))
        scored = CandidateEvidenceScorer().score_all([weak, strong], list(_chunks().values()), LOOKUP)
        ranked = CandidateRanker(diversity_penalty=0).rank(scored, top_k=2)
        assert ranked[0].gap_statement.startswith("zz strong")
        # 2 extra gap terms: relevance c1 4/6, c2 2/6 → mean .5;
        # support = .35·.5 + .2·.75 + .15·.5 + .3·(2/3) = .6
        assert ranked[0].metadata["ranking"]["evidence_score"] == pytest.approx(0.6, abs=1e-6)
        assert ranked[1].metadata["ranking"]["evidence_score"] < 0.6

    def test_insufficient_evidence_ranks_below_supported(self):
        # Identical heuristic inputs (confidence, 2 cited passages, no source papers);
        # the unsupported candidate sorts first alphabetically (adversarial tie-break).
        supported = _gap(statement="zz supported alpha beta gamma delta", initial_confidence=0.6)
        unsupported = _gap(statement="aa unsupported alpha beta gamma delta",
                           supporting=("ghost-1", "ghost-2"), initial_confidence=0.6)
        ranker = CandidateRanker(diversity_penalty=0, evidence_weight=0)
        assert ranker.base_score(supported) == ranker.base_score(unsupported)
        scored = CandidateEvidenceScorer().score_all([unsupported, supported],
                                                     list(_chunks().values()), LOOKUP)
        ranked = CandidateRanker(diversity_penalty=0).rank(scored, top_k=2)
        assert ranked[0].gap_statement.startswith("zz supported")
        assert ranked[1].evidence_assessment.status == (
            CandidateEvidenceStatus.INSUFFICIENT_EVIDENCE
        )


# ======================================================================
# Review fix 1 + 4: shared reference year, reference-year validation
# ======================================================================


def _dated(cid: str, pid: str, year) -> PaperChunk:
    meta = {} if year is None else {"publication_year": year}
    return _chunk(cid, pid, "Abstract", "alpha beta gamma delta", year=year)


class TestSharedReferenceYear:
    """
    Old candidate A (two 1995 papers) vs newer candidate B (2024 + 2014 papers).
    Statements "alpha beta gamma delta one|two": 5 gap terms, 4 in each passage
    → relevance .8 for both; both have 2 independent papers (independence 2/3),
    so only recency differs.  Shared reference year = 2024 (latest across all
    supplied chunks):
        A: relevance .8, quality .5, recency 0,  independence 2/3 → .28+.1+0+.2    = .58
        B: relevance .8, quality .5, recency .5, independence 2/3 → .28+.1+.075+.2 = .655
    With the old per-candidate reference year A's recency was 1.0 → .73, so A
    wrongly outranked B.
    """

    CHUNKS = [_dated("old", "p-old", 1995), _dated("old2", "p-old2", 1995),
              _dated("new", "p-new", 2024), _dated("mid", "p-mid", 2014)]

    def _pair(self):
        a = _gap(statement=STATEMENT + " one", supporting=("old", "old2"), initial_confidence=0.9)
        b = _gap(statement=STATEMENT + " two", supporting=("new", "mid"), initial_confidence=0.5)
        return a, b

    def test_old_candidate_scored_against_shared_year(self):
        a, b = CandidateEvidenceScorer().score_all(list(self._pair()), self.CHUNKS, LOOKUP)
        assert a.evidence_assessment.features["recency"] == 0.0
        assert a.evidence_assessment.score == pytest.approx(0.58, abs=1e-6)
        assert b.evidence_assessment.features["recency"] == pytest.approx(0.5)
        assert b.evidence_assessment.score == pytest.approx(0.655, abs=1e-6)

    def test_newer_multi_year_candidate_ranks_first(self):
        # A wins on the heuristic (confidence .9 vs .5); with evidence_weight=1.0
        # the ranking follows the evidence score alone, which favours B.
        scored = CandidateEvidenceScorer().score_all(list(self._pair()), self.CHUNKS, LOOKUP)
        ranked = CandidateRanker(diversity_penalty=0, evidence_weight=1.0).rank(scored, 2)
        assert [g.relevant_passages for g in ranked] == [["new", "mid"], ["old", "old2"]]

    def test_single_score_call_uses_all_supplied_chunks(self):
        a, _ = self._pair()
        alone = CandidateEvidenceScorer().score(a, {c.chunk_id: c for c in self.CHUNKS}, LOOKUP)
        assert alone.supporting[0].recency == 0.0  # not 1.0 from its own 1995 paper

    def test_resolve_reference_year_automatic(self):
        scorer = CandidateEvidenceScorer()
        assert scorer.resolve_reference_year(self.CHUNKS, LOOKUP) == 2024
        assert scorer.resolve_reference_year([_dated("x", "p", None)], LOOKUP) is None
        assert scorer.resolve_reference_year([]) is None

    def test_resolve_reference_year_uses_paper_lookup(self):
        store = make_sample_corpus()  # chunks carry no year; papers do (2017–2021)
        assert CandidateEvidenceScorer().resolve_reference_year(
            store.get_all_chunks(), store.get_paper_by_id
        ) == 2021

    def test_explicit_reference_year_overrides(self):
        scorer = CandidateEvidenceScorer(reference_year=2030)
        assert scorer.resolve_reference_year(self.CHUNKS, LOOKUP) == 2030
        a, _ = scorer.score_all(list(self._pair()), self.CHUNKS, LOOKUP)
        assert a.evidence_assessment.supporting[0].recency == 0.0  # 35 years > horizon
        b = CandidateEvidenceScorer(reference_year=2024).score_all(list(self._pair()), self.CHUNKS, LOOKUP)[1]
        assert [i.recency for i in b.evidence_assessment.supporting] == pytest.approx([1.0, 0.0])

    def test_missing_year_still_excluded(self):
        gap = _gap(supporting=("undated",))
        [scored] = CandidateEvidenceScorer().score_all(
            [gap], self.CHUNKS + [_dated("undated", "p-u", None)], LOOKUP
        )
        assert scored.evidence_assessment.supporting[0].recency is None
        assert scored.evidence_assessment.features["recency"] is None

    @pytest.mark.parametrize(
        ("paper_year", "expected"),
        [(2020, 1.0), (2017, 0.7), (2010, 0.0), (1990, 0.0), (2026, 1.0)],
        ids=["same_year", "older", "horizon", "beyond_horizon", "newer_capped"],
    )
    def test_recency_formula_matches_docs(self, paper_year, expected):
        """recency = max(0, 1 − max(0, reference_year − year) / 10), reference 2020."""
        chunk = _dated("x", "p", paper_year)
        a = CandidateEvidenceScorer(reference_year=2020).score(
            _gap(supporting=("x",)), {"x": chunk}, LOOKUP
        )
        assert a.supporting[0].recency == pytest.approx(expected)

    @pytest.mark.parametrize("year", [0, -5, True, 2020.5, "2020"])
    def test_invalid_explicit_reference_year(self, year):
        with pytest.raises(ValueError, match="reference_year"):
            CandidateEvidenceScorer(reference_year=year)


# ======================================================================
# Review fix 2: zero-strength counter evidence
# ======================================================================


class TestZeroStrengthCounterEvidence:
    def _chunks(self):
        return {
            "s": _chunk("s", "p1", "Limitations", "alpha beta gamma delta"),
            "z": _chunk("z", "p2", "Results", "completely unrelated words"),
        }

    def test_zero_strength_counter_does_not_contest(self):
        gap = _gap(supporting=("s",), contradicting=("z",))
        a = CandidateEvidenceScorer().score(gap, self._chunks(), LOOKUP)
        assert a.contradicting[0].strength == 0.0
        assert a.contradiction_ratio == 0.0
        assert a.score == a.support_score
        assert a.status == CandidateEvidenceStatus.SUPPORTED
        assert "1 contradicting passage(s) had zero strength" in a.explanation
        assert "reduce the score" not in a.explanation

    def test_zero_strength_counter_with_no_support(self):
        gap = _gap(supporting=(), contradicting=("z",))
        a = CandidateEvidenceScorer().score(gap, self._chunks(), LOOKUP)
        assert a.status == CandidateEvidenceStatus.INSUFFICIENT_EVIDENCE
        assert a.contradiction_ratio == 0.0
        # Explanation must not claim the zero-strength passage counts against the gap.
        assert "1 contradicting passage(s) had zero strength" in a.explanation
        assert "did not affect the assessment" in a.explanation
        assert "suggest the gap may already be addressed" not in a.explanation

    def test_mixed_counter_passages_with_no_support(self):
        chunks = {**self._chunks(), "c": _chunk("c", "p3", "Results", "alpha beta")}
        a = CandidateEvidenceScorer().score(_gap(supporting=(), contradicting=("z", "c")), chunks, LOOKUP)
        assert a.status == CandidateEvidenceStatus.INSUFFICIENT_EVIDENCE
        assert a.contradiction_ratio == 1.0
        assert "1 passage(s) suggest the gap may already be addressed" in a.explanation
        assert "1 contradicting passage(s) had zero strength" in a.explanation

    def test_effective_and_ineffective_counter_passages(self):
        chunks = {**self._chunks(), "c": _chunk("c", "p3", "Results", "alpha beta")}
        gap = _gap(supporting=("s",), contradicting=("z", "c"))
        a = CandidateEvidenceScorer().score(gap, chunks, LOOKUP)
        assert a.status == CandidateEvidenceStatus.CONTESTED
        assert a.contradiction_ratio > 0
        assert "1 contradicting passage(s) carry" in a.explanation
        assert "1 contradicting passage(s) had zero strength" in a.explanation


# ======================================================================
# Review fix 3: passage-level weights
# ======================================================================


class TestPassageLevelWeights:
    @pytest.mark.parametrize(
        "weights",
        [{"independence": 1.0}, {"quality": 1.0}, {"recency": 1.0},
         {"relevance": 0.0, "quality": 1.0, "recency": 1.0, "independence": 1.0}],
    )
    def test_relevance_weight_required(self, weights):
        with pytest.raises(ValueError, match="relevance"):
            CandidateEvidenceScorer(weights=weights)

    def test_counter_evidence_always_measurable(self):
        """With relevance weighted, a counter passage sharing gap terms has strength > 0."""
        a = CandidateEvidenceScorer(weights={"relevance": 0.1, "independence": 1.0}).score(
            _gap(contradicting=("c3",)), _chunks(), LOOKUP
        )
        assert a.contradicting[0].strength > 0
        assert a.contradiction_ratio > 0
