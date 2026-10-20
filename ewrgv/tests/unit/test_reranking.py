from __future__ import annotations

import sys

import numpy as np
import pytest

from app.core.config import settings
from app.core.exceptions import RetrievalError
from app.domain.models.retrieval_result import RetrievalResult
from app.retrieval.reranking import CrossEncoderReranker, RerankerError
from app.retrieval.service import RetrievalService, _cap_per_paper
from tests.fixtures.sample_corpus import make_sample_corpus


class FakeCrossEncoder:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[list[tuple[str, str]], dict]] = []

    def predict(self, pairs, **kwargs):
        self.calls.append((pairs, kwargs))
        return np.asarray(self.scores[:len(pairs)])


def _result(chunk_id: str, score: float, rank: int, text: str = "passage") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id, paper_id=f"paper-{chunk_id}", text=text,
        score=score, rank=rank, retrieval_method="sparse",
    )


def test_cross_encoder_scores_pairs_and_preserves_original_score() -> None:
    model = FakeCrossEncoder([0.1, 3.2])
    reranker = CrossEncoderReranker(model=model)
    original = [_result("a", 8.5, 1, "first"), _result("b", 4.25, 2, "second")]

    ranked, elapsed = reranker.rerank_results("research question", original, top_n=2)

    assert [result.chunk_id for result in ranked] == ["b", "a"]
    assert [result.score for result in ranked] == [4.25, 8.5]
    assert [result.metadata["reranker_score"] for result in ranked] == [3.2, 0.1]
    assert ranked[0].metadata["retrieval_rank"] == 2
    assert model.calls[0][0] == [("research question", "first"), ("research question", "second")]
    assert model.calls[0][1]["show_progress_bar"] is False
    assert elapsed >= 0


def test_empty_input_does_not_load_or_call_model() -> None:
    model = FakeCrossEncoder([])
    reranker = CrossEncoderReranker(model=model)
    assert reranker.score_pairs("q", []) == []
    assert model.calls == []


def test_model_is_loaded_only_on_first_nonempty_request(monkeypatch) -> None:
    model = FakeCrossEncoder([0.6])
    constructed = []

    def factory(name, device):
        constructed.append((name, device))
        return model

    monkeypatch.setitem(sys.modules, "sentence_transformers", type("Module", (), {
        "CrossEncoder": staticmethod(factory),
    }))
    reranker = CrossEncoderReranker(model_name="local/model", device="cpu")

    assert reranker.score_pairs("q", []) == []
    assert constructed == []
    assert reranker.score_pairs("q", ["p"]) == [0.6]
    assert constructed == [("local/model", "cpu")]
    reranker.score_pairs("q", ["again"])
    assert len(constructed) == 1


def test_invalid_model_scores_are_rejected() -> None:
    reranker = CrossEncoderReranker(model=FakeCrossEncoder([float("nan")]))
    with pytest.raises(RerankerError, match="invalid scores"):
        reranker.score_pairs("q", ["passage"])


def test_service_reranks_when_enabled_and_keeps_fallback_status(monkeypatch) -> None:
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)
    monkeypatch.setattr(settings, "RERANK_TOP_N", 2)
    monkeypatch.setattr(settings, "RERANK_MAX_CANDIDATES", 2)
    reranker = CrossEncoderReranker(model=FakeCrossEncoder([0.2, 0.9]))
    service = RetrievalService(make_sample_corpus(), reranker=reranker)
    originals = [_result("a", 7.0, 1), _result("b", 5.0, 2), _result("c", 2.0, 3)]
    monkeypatch.setattr(service, "_ranked", lambda *args: originals)

    ranked = service.retrieve("question", top_k=3, method="sparse")

    assert [result.chunk_id for result in ranked] == ["b", "a", "c"]
    assert [result.score for result in ranked] == [5.0, 7.0, 2.0]
    assert ranked[0].metadata["reranking"]["status"] == "applied"
    assert "reranker_score" not in ranked[2].metadata


def test_service_fails_open_with_status(monkeypatch) -> None:
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)
    monkeypatch.setattr(settings, "RERANK_FAIL_OPEN", True)
    reranker = CrossEncoderReranker(model=FakeCrossEncoder([float("nan")]))
    service = RetrievalService(make_sample_corpus(), reranker=reranker)
    originals = [_result("a", 1.0, 1)]
    monkeypatch.setattr(service, "_ranked", lambda *args: originals)

    ranked = service.retrieve("question", top_k=1, method="sparse")

    assert ranked[0].chunk_id == "a"
    assert ranked[0].metadata["reranking"]["status"] == "failed_fallback"
    assert ranked[0].metadata["reranking"]["error_type"] == "RerankerError"


def test_service_can_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)
    monkeypatch.setattr(settings, "RERANK_FAIL_OPEN", False)
    reranker = CrossEncoderReranker(model=FakeCrossEncoder([float("nan")]))
    service = RetrievalService(make_sample_corpus(), reranker=reranker)
    monkeypatch.setattr(service, "_ranked", lambda *args: [_result("a", 1.0, 1)])

    with pytest.raises(RetrievalError, match="RERANK_FAIL_OPEN"):
        service.retrieve("question", top_k=1, method="sparse")


def test_per_paper_backfill_keeps_reranked_order() -> None:
    results = [
        _result("a1", 0.1, 1).model_copy(update={
            "metadata": {"reranker_score": 3.0, "reranking": {"status": "applied"}}
        }),
        _result("a2", 0.9, 2).model_copy(update={
            "metadata": {"reranker_score": 2.0, "reranking": {"status": "applied"}}
        }),
        _result("b1", 0.8, 3).model_copy(update={
            "metadata": {"reranker_score": 1.0, "reranking": {"status": "applied"}}
        }),
    ]

    capped = _cap_per_paper(results, top_k=3, cap=1)

    assert [result.chunk_id for result in capped] == ["a1", "a2", "b1"]


def test_multiquery_reranks_each_passage_against_its_best_query(monkeypatch) -> None:
    monkeypatch.setattr(settings, "RERANK_ENABLED", True)
    monkeypatch.setattr(settings, "RERANK_TOP_N", 3)
    monkeypatch.setattr(settings, "RERANK_MAX_CANDIDATES", 3)
    monkeypatch.setattr(settings, "RETRIEVAL_MAX_PER_PAPER", 0)
    model = FakeCrossEncoder([0.1, 0.2, 0.9])
    service = RetrievalService(make_sample_corpus(), reranker=CrossEncoderReranker(model=model))
    results = [
        _result("b", 2.0, 1, "passage-b"),
        _result("a", 1.0, 2, "passage-a"),
        _result("c", 0.5, 3, "passage-c"),
    ]
    queries = {"a": "q1", "b": "q2", "c": "q2"}
    ranked = service._rerank_results(results, queries)

    assert [result.chunk_id for result in ranked] == ["c", "a", "b"]
    assert ranked[0].score == 0.5
    assert model.calls[0][0] == [
        ("q2", "passage-b"), ("q1", "passage-a"), ("q2", "passage-c")
    ]
