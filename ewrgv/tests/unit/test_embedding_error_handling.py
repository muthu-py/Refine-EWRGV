"""
tests/unit/test_embedding_error_handling.py
---------------------------------------------
Embedding API failure handling end to end:

* provider: 429 retried (Retry-After / Gemini retryDelay honoured, capped),
  typed EmbeddingAPIError for 401 / 400 / 429-exhausted
* API: rate limit -> 429 + Retry-After header, auth / bad request -> 502
* /represent: a rate limit stops embedding early and is reported on the job
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

import app.api.routes.research as research_routes
from app.core.config import settings
from app.core.exceptions import EmbeddingAPIError
from app.core.job_store import RepresentationJob
from app.domain.models.paper_chunk import ChunkingDiagnostics, ChunkingResult, PaperChunk
from app.ingestion.representation.service import RepresentationService
from app.main import app
from app.providers.embeddings.openai_embeddings import OpenAIEmbeddingProvider
from app.retrieval.service import RetrievalService
from tests.fixtures.sample_corpus import FakeEmbeddingProvider, make_sample_corpus

GEMINI_429 = [{
    "error": {
        "code": 429,
        "message": "Resource has been exhausted (e.g. check quota).",
        "status": "RESOURCE_EXHAUSTED",
        "details": [
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "37s"}
        ],
    }
}]
GEMINI_400 = {
    "error": {
        "code": 400,
        "message": "* BatchEmbedContentsRequest.requests: at most 100 requests can be in one batch\n",
        "status": "INVALID_ARGUMENT",
    }
}


def _response(status: int, body=None, headers: dict | None = None) -> MagicMock:
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status
    if body is None:
        resp.json.side_effect = ValueError("no json")
        resp.text = ""
    else:
        resp.json.return_value = body
        resp.text = json.dumps(body)
    resp.headers = headers or {}
    return resp


def _ok(n: int = 1, dim: int = 4) -> MagicMock:
    return _response(200, {"data": [{"index": i, "embedding": [0.1] * dim} for i in range(n)]})


@pytest.fixture
def provider() -> OpenAIEmbeddingProvider:
    return OpenAIEmbeddingProvider(
        api_key="sk-test", model="m", dimension=4, max_retries=2, retry_delay=0.01
    )


@pytest.fixture
def sleeps():
    with patch(
        "app.providers.embeddings.openai_embeddings.asyncio.sleep", new_callable=AsyncMock
    ) as sleep:
        yield sleep


# ------------------------------------------------------------------ #
# Provider
# ------------------------------------------------------------------ #


@pytest.mark.asyncio
async def test_429_then_200_succeeds_after_gemini_retry_delay(provider, sleeps) -> None:
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as post:
        post.side_effect = [_response(429, GEMINI_429), _ok()]
        vectors = await provider.embed_documents(["x"])
    assert len(vectors) == 1
    assert post.call_count == 2
    sleeps.assert_awaited_once_with(37.0)


@pytest.mark.asyncio
async def test_retry_after_header_is_capped(provider, sleeps, monkeypatch) -> None:
    monkeypatch.setattr(provider, "_max_backoff", 5.0)
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as post:
        post.side_effect = [_response(429, {}, headers={"Retry-After": "120"}), _ok()]
        await provider.embed_documents(["x"])
    sleeps.assert_awaited_once_with(5.0)


@pytest.mark.asyncio
async def test_backoff_without_hint_grows_and_is_capped(provider, sleeps, monkeypatch) -> None:
    monkeypatch.setattr(provider, "_max_backoff", 0.015)
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as post:
        post.side_effect = [_response(503), _response(503), _ok()]
        await provider.embed_documents(["x"])
    waits = [c.args[0] for c in sleeps.await_args_list]
    assert len(waits) == 2
    assert 0.01 <= waits[0] <= 0.015
    assert waits[1] == 0.015


@pytest.mark.asyncio
async def test_429_exhausted_raises_rate_limit_error(provider, sleeps) -> None:
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as post:
        post.return_value = _response(429, GEMINI_429)
        with pytest.raises(EmbeddingAPIError) as err:
            await provider.embed_documents(["x"])
    assert post.call_count == 3  # max_retries=2 -> 3 attempts
    assert err.value.kind == "rate_limit"
    assert err.value.retry_after == 37.0
    assert "retry in 37s" in err.value.message
    assert "Resource has been exhausted" in err.value.message


@pytest.mark.asyncio
async def test_401_is_auth_error_not_retried(provider, sleeps) -> None:
    body = {"error": {"message": "Incorrect API key provided"}}
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as post:
        post.return_value = _response(401, body)
        with pytest.raises(EmbeddingAPIError) as err:
            await provider.embed_documents(["x"])
    assert post.call_count == 1
    assert err.value.kind == "auth"
    assert "EMBEDDING_API_KEY" in err.value.message
    assert "sk-test" not in err.value.message


@pytest.mark.asyncio
async def test_400_surfaces_provider_message(provider, sleeps) -> None:
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as post:
        post.return_value = _response(400, GEMINI_400)
        with pytest.raises(EmbeddingAPIError) as err:
            await provider.embed_documents(["x"])
    assert post.call_count == 1
    assert err.value.kind == "bad_request"
    assert err.value.provider_message == (
        "BatchEmbedContentsRequest.requests: at most 100 requests can be in one batch"
    )


# ------------------------------------------------------------------ #
# API mapping
# ------------------------------------------------------------------ #


class _FailingEmbedder(FakeEmbeddingProvider):
    def __init__(self, error: EmbeddingAPIError) -> None:
        super().__init__()
        self._error = error

    async def embed_documents(self, texts):
        raise self._error


def _serve(monkeypatch, error: EmbeddingAPIError) -> None:
    async def fake_build(run_id: str, with_dense: bool) -> RetrievalService:
        return RetrievalService(make_sample_corpus(), _FailingEmbedder(error))

    monkeypatch.setattr(research_routes, "_build_retrieval_service", fake_build)


@pytest.mark.parametrize("method", ["dense", "hybrid"])
def test_rate_limit_returns_429_with_retry_after(monkeypatch, method) -> None:
    _serve(monkeypatch, EmbeddingAPIError(429, "quota exhausted", retry_after=36.2))
    resp = TestClient(app).post(
        f"{settings.API_V1_PREFIX}/research/retrieve/{method}",
        json={"run_id": "r1", "query": "attention"},
    )
    assert resp.status_code == 429
    assert resp.headers["Retry-After"] == "37"
    assert "rate limit" in resp.json()["error"]
    assert resp.json()["detail"] == "quota exhausted"


@pytest.mark.parametrize("status", [401, 400])
def test_auth_and_bad_request_return_502(monkeypatch, status) -> None:
    _serve(monkeypatch, EmbeddingAPIError(status, "upstream says no"))
    resp = TestClient(app).post(
        f"{settings.API_V1_PREFIX}/research/retrieve/dense",
        json={"run_id": "r1", "query": "attention"},
    )
    assert resp.status_code == 502
    assert resp.json()["detail"] == "upstream says no"


# ------------------------------------------------------------------ #
# /represent
# ------------------------------------------------------------------ #


def _chunks_json(paper_id: str, texts: list[str]) -> bytes:
    chunks = [
        PaperChunk(chunk_id=f"{paper_id}-c{i}", paper_id=paper_id, section_id="s",
                   section_title="T", text=t)
        for i, t in enumerate(texts)
    ]
    return ChunkingResult(
        paper_id=paper_id, source_sections_path="x", chunks=chunks,
        diagnostics=ChunkingDiagnostics(),
    ).model_dump_json().encode("utf-8")


@pytest.mark.asyncio
async def test_represent_stops_on_rate_limit_and_reports_it(monkeypatch) -> None:
    monkeypatch.setattr(settings, "EMBEDDING_BATCH_SIZE", 1)
    provider = AsyncMock()
    provider.get_model_name = MagicMock(return_value="m")
    provider.get_dimension = MagicMock(return_value=3)
    provider.embed_documents.side_effect = [
        [[0.1, 0.2, 0.3]],
        EmbeddingAPIError(429, "quota", retry_after=30),
    ]
    embedding_repo = AsyncMock()
    embedding_repo.get_embedded_chunk_ids.return_value = set()
    embedding_repo.get_embedding_counts.return_value = 1
    paper_repo = AsyncMock()
    paper_repo.get_papers_for_representation.return_value = [
        {"paper_id": "p1"}, {"paper_id": "p2"}
    ]
    storage = AsyncMock()
    storage.download.side_effect = [
        _chunks_json("p1", ["a", "b", "   "]),  # blank chunk is never sent
        _chunks_json("p2", ["c"]),
    ]
    job = RepresentationJob(job_id="j", run_id="r")
    service = RepresentationService(provider, embedding_repo, paper_repo, storage)

    with patch("app.ingestion.representation.service.BM25Index.save"):
        await service.run_representation_for_run("r", job)

    sent = [c.args[0] for c in provider.embed_documents.await_args_list]
    assert sent == [["a"], ["b"]]  # stopped after the 429; p2 never sent
    assert job.embedded_chunks == 1
    assert job.failed_chunks == 2
    assert job.embedding_status == "PARTIAL"
    assert "rate limit" in job.error and "re-run /represent" in job.error
