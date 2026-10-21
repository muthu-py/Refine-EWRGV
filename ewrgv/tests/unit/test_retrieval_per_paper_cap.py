"""
tests/unit/test_retrieval_per_paper_cap.py
--------------------------------------------
RetrievalService caps hits per paper (RETRIEVAL_MAX_PER_PAPER) so a single
paper cannot fill the top-k list.
"""

from __future__ import annotations

from collections import Counter

from app.core.config import settings
from app.domain.models.retrieval_result import RetrievalResult
from app.retrieval.service import RetrievalService, _cap_per_paper
from tests.fixtures.sample_corpus import FakeEmbeddingProvider, make_sample_corpus


def _r(paper: str, score: float, i: int) -> RetrievalResult:
    return RetrievalResult(chunk_id=f"{paper}-{i}", paper_id=paper, text="t", score=score,
                           rank=i, retrieval_method="dense")


def test_cap_keeps_order_and_limits_each_paper() -> None:
    results = [_r("A", 0.9 - i * 0.01, i) for i in range(6)] + [_r("B", 0.5, 10), _r("C", 0.4, 11)]
    out = _cap_per_paper(results, top_k=4, cap=2)
    assert [r.paper_id for r in out] == ["A", "A", "B", "C"]
    assert [r.rank for r in out] == [1, 2, 3, 4]


def test_backfills_when_too_few_papers() -> None:
    results = [_r("A", 0.9 - i * 0.01, i) for i in range(5)] + [_r("B", 0.5, 10)]
    out = _cap_per_paper(results, top_k=5, cap=2)
    assert len(out) == 5
    assert [r.score for r in out] == sorted((r.score for r in out), reverse=True)


def test_cap_zero_disables() -> None:
    results = [_r("A", 0.9 - i * 0.01, i) for i in range(5)]
    assert [r.chunk_id for r in _cap_per_paper(results, 3, 0)] == ["A-0", "A-1", "A-2"]


def test_service_applies_cap(monkeypatch) -> None:
    monkeypatch.setattr(settings, "RETRIEVAL_MAX_PER_PAPER", 1)
    service = RetrievalService(make_sample_corpus(), FakeEmbeddingProvider())
    for method in ("sparse", "dense", "hybrid"):
        results = service.retrieve("language model attention", top_k=3, method=method)
        papers = Counter(r.paper_id for r in results)
        if len(papers) == len(results):
            assert max(papers.values()) == 1
        assert [r.rank for r in results] == list(range(1, len(results) + 1))
