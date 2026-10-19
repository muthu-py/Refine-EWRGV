"""
tests/unit/test_retrieval_service_batching.py
-----------------------------------------------
RetrievalService.aretrieve must embed the query + chunk texts in batches of
at most EMBEDDING_BATCH_SIZE (Gemini rejects more than 100 inputs per request)
and keep vectors aligned with their texts.
"""

from __future__ import annotations

import asyncio

from app.core.config import settings
from app.retrieval.service import RetrievalService
from tests.fixtures.sample_corpus import FakeEmbeddingProvider, make_sample_corpus


class RecordingEmbedder(FakeEmbeddingProvider):
    def __init__(self) -> None:
        super().__init__()
        self.async_batches: list[list[str]] = []

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.async_batches.append(list(texts))
        return await super().embed_documents(texts)


def _ids(results) -> list[tuple[str, float]]:
    return [(r.chunk_id, round(r.score, 9)) for r in results]


def test_aretrieve_embeds_in_bounded_batches(monkeypatch) -> None:
    monkeypatch.setattr(settings, "EMBEDDING_BATCH_SIZE", 3)
    embedder = RecordingEmbedder()
    service = RetrievalService(make_sample_corpus(), embedder)

    asyncio.run(service.aretrieve("transformer attention", top_k=5, method="dense"))

    assert len(embedder.async_batches) > 1
    assert all(0 < len(b) <= 3 for b in embedder.async_batches)
    sent = [t for b in embedder.async_batches for t in b]
    assert sent[0] == "transformer attention"
    assert len(sent) == len(set(sent))


def test_batched_results_match_single_batch(monkeypatch) -> None:
    monkeypatch.setattr(settings, "EMBEDDING_BATCH_SIZE", 1000)
    single = asyncio.run(
        RetrievalService(make_sample_corpus(), RecordingEmbedder()).aretrieve(
            "language model", top_k=10, method="hybrid"
        )
    )
    monkeypatch.setattr(settings, "EMBEDDING_BATCH_SIZE", 2)
    batched = asyncio.run(
        RetrievalService(make_sample_corpus(), RecordingEmbedder()).aretrieve(
            "language model", top_k=10, method="hybrid"
        )
    )
    assert _ids(batched) == _ids(single)
