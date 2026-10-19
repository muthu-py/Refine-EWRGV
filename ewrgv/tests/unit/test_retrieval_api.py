"""
tests/unit/test_retrieval_api.py
----------------------------------
Tests for the three retrieval API endpoints:

    POST /api/v1/research/retrieve/dense
    POST /api/v1/research/retrieve/sparse
    POST /api/v1/research/retrieve/hybrid

Each request names a ``run_id``; the endpoint loads that run's RunCorpus
(PaperChunks from chunks.json, stored embeddings, BM25 index).  Endpoint
tests patch the corpus loading with the in-memory sample corpus and a
deterministic embedder; TestBuildRetrievalService covers the loading itself
with mocked repositories (no database, storage or API key).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import app.api.routes.research as research_routes
from app.main import app
from app.retrieval.service import RetrievalService
from tests.fixtures.sample_corpus import FakeEmbeddingProvider, make_sample_corpus

RUN = "run-sample"
_REAL_BUILD = research_routes._build_retrieval_service  # before the autouse patch


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def sample_run(monkeypatch):
    """Serve every run_id from the sample corpus; records the calls."""
    calls: list[tuple[str, bool]] = []

    async def fake_build(run_id: str, with_dense: bool) -> RetrievalService:
        calls.append((run_id, with_dense))
        return RetrievalService(make_sample_corpus(), FakeEmbeddingProvider())

    monkeypatch.setattr(research_routes, "_build_retrieval_service", fake_build)
    return calls


# ====================================================================
# Dense: POST /retrieve/dense
# ====================================================================


class TestRetrieveDense:
    """POST /api/v1/research/retrieve/dense."""

    URL = "/api/v1/research/retrieve/dense"

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"run_id": RUN, "query": "Transformer attention", "top_k": 5})
        assert resp.status_code == 200

    def test_response_schema(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"run_id": RUN, "query": "Transformer attention", "top_k": 3})
        body = resp.json()
        assert body["query"] == "Transformer attention"
        assert body["run_id"] == RUN
        assert body["top_k"] == 3
        assert "total_results" in body
        assert isinstance(body["results"], list)
        assert len(body["results"]) > 0

    def test_result_items_have_required_fields(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"run_id": RUN, "query": "neural network", "top_k": 3})
        for item in resp.json()["results"]:
            assert "chunk_id" in item
            assert "paper_id" in item
            assert "section" in item
            assert "text" in item
            assert "score" in item
            assert "rank" in item
            assert "retrieval_method" in item
            assert "section_type" in item
            assert "page_start" in item and "page_end" in item

    def test_retrieval_method_is_dense(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"run_id": RUN, "query": "attention", "top_k": 3})
        for item in resp.json()["results"]:
            assert item["retrieval_method"] == "dense"

    def test_respects_top_k(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"run_id": RUN, "query": "model", "top_k": 3})
        assert len(resp.json()["results"]) <= 3


# ====================================================================
# Sparse: POST /retrieve/sparse
# ====================================================================


class TestRetrieveSparse:
    """POST /api/v1/research/retrieve/sparse."""

    URL = "/api/v1/research/retrieve/sparse"

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"run_id": RUN, "query": "Transformer attention", "top_k": 5})
        assert resp.status_code == 200

    def test_retrieval_method_is_sparse(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"run_id": RUN, "query": "language model", "top_k": 3})
        for item in resp.json()["results"]:
            assert item["retrieval_method"] == "sparse"

    def test_respects_top_k(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"run_id": RUN, "query": "Transformer", "top_k": 2})
        assert len(resp.json()["results"]) <= 2


# ====================================================================
# Hybrid: POST /retrieve/hybrid
# ====================================================================


class TestRetrieveHybrid:
    """POST /api/v1/research/retrieve/hybrid."""

    URL = "/api/v1/research/retrieve/hybrid"

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"run_id": RUN, "query": "Transformer attention", "top_k": 5})
        assert resp.status_code == 200

    def test_retrieval_method_is_hybrid(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"run_id": RUN, "query": "language model", "top_k": 3})
        for item in resp.json()["results"]:
            assert item["retrieval_method"] == "hybrid"

    def test_no_duplicate_chunks(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"run_id": RUN, "query": "Transformer model", "top_k": 20})
        ids = [r["chunk_id"] for r in resp.json()["results"]]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs in hybrid results"

    def test_provenance_metadata(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"run_id": RUN, "query": "Transformer", "top_k": 5})
        for item in resp.json()["results"]:
            assert "contributing_methods" in item["metadata"]

    def test_respects_top_k(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"run_id": RUN, "query": "Transformer", "top_k": 3})
        assert len(resp.json()["results"]) <= 3


# ====================================================================
# Default top_k
# ====================================================================


class TestRetrieveDefaults:
    """Default parameter behavior across all endpoints."""

    def test_default_top_k_is_10(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/research/retrieve/hybrid",
            json={"run_id": RUN, "query": "language model"},
        )
        assert resp.json()["top_k"] == 10


# ====================================================================
# Validation — shared across all endpoints
# ====================================================================


class TestRetrieveValidation:
    """Validation / error handling (tested against /retrieve/dense as representative)."""

    URL = "/api/v1/research/retrieve/dense"

    def test_empty_query_returns_400(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"run_id": RUN, "query": ""})
        assert resp.status_code == 400

    def test_whitespace_query_returns_400(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"run_id": RUN, "query": "   "})
        assert resp.status_code == 400

    def test_missing_query_returns_422(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"run_id": RUN, "top_k": 5})
        assert resp.status_code == 422

    def test_missing_run_id_returns_422(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"query": "test"})
        assert resp.status_code == 422

    def test_blank_run_id_returns_400(self, client: TestClient, sample_run) -> None:
        resp = client.post(self.URL, json={"run_id": "  ", "query": "test"})
        assert resp.status_code == 400
        assert sample_run == []  # rejected before any corpus is loaded

    def test_top_k_zero_returns_400(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"run_id": RUN, "query": "test", "top_k": 0})
        assert resp.status_code == 400

    def test_top_k_negative_returns_400(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"run_id": RUN, "query": "test", "top_k": -5})
        assert resp.status_code == 400

    def test_top_k_too_large_returns_400(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"run_id": RUN, "query": "test", "top_k": 999})
        assert resp.status_code == 400

    def test_validation_on_sparse(self, client: TestClient) -> None:
        """Validation works on /sparse too."""
        resp = client.post("/api/v1/research/retrieve/sparse", json={"run_id": RUN, "query": ""})
        assert resp.status_code == 400

    def test_validation_on_hybrid(self, client: TestClient) -> None:
        """Validation works on /hybrid too."""
        resp = client.post("/api/v1/research/retrieve/hybrid", json={"run_id": RUN, "query": ""})
        assert resp.status_code == 400


# ====================================================================
# Old /retrieve endpoint is removed
# ====================================================================


class TestOldEndpointRemoved:
    """The generic /retrieve endpoint should no longer exist."""

    def test_old_retrieve_returns_404_or_405(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/research/retrieve",
            json={"run_id": RUN, "query": "test", "method": "dense"},
        )
        # Should not be 200 — it's been removed
        assert resp.status_code in (404, 405, 307)


# ====================================================================
# Determinism
# ====================================================================


class TestRetrieveDeterminism:
    """Results are deterministic for the same input."""

    def test_dense_deterministic(self, client: TestClient) -> None:
        payload = {"run_id": RUN, "query": "BERT pre-training", "top_k": 5}
        r1 = client.post("/api/v1/research/retrieve/dense", json=payload).json()
        r2 = client.post("/api/v1/research/retrieve/dense", json=payload).json()
        assert [r["chunk_id"] for r in r1["results"]] == [r["chunk_id"] for r in r2["results"]]

    def test_hybrid_deterministic(self, client: TestClient) -> None:
        payload = {"run_id": RUN, "query": "BERT pre-training", "top_k": 5}
        r1 = client.post("/api/v1/research/retrieve/hybrid", json=payload).json()
        r2 = client.post("/api/v1/research/retrieve/hybrid", json=payload).json()
        assert [r["chunk_id"] for r in r1["results"]] == [r["chunk_id"] for r in r2["results"]]


# ====================================================================
# Ranks
# ====================================================================


class TestRetrieveRanks:
    """Rank ordering validation."""

    def test_ranks_are_sequential_dense(self, client: TestClient) -> None:
        resp = client.post("/api/v1/research/retrieve/dense", json={"run_id": RUN, "query": "attention", "top_k": 5})
        for i, r in enumerate(resp.json()["results"]):
            assert r["rank"] == i + 1

    def test_ranks_are_sequential_hybrid(self, client: TestClient) -> None:
        resp = client.post("/api/v1/research/retrieve/hybrid", json={"run_id": RUN, "query": "attention", "top_k": 5})
        for i, r in enumerate(resp.json()["results"]):
            assert r["rank"] == i + 1


# ====================================================================
# Run corpus: provenance and corpus loading
# ====================================================================


class TestRunCorpusResults:
    def test_results_carry_paper_chunk_provenance(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/research/retrieve/sparse",
            json={"run_id": RUN, "query": "GPT-3 limitations", "top_k": 1},
        )
        [item] = resp.json()["results"]
        assert item["chunk_id"] == "chunk-003-limits"
        assert item["section"] == "Limitations"
        assert item["section_type"] == "LIMITATIONS"
        assert item["page_start"] >= 1
        assert item["metadata"]["publication_year"] == 2020

    def test_sparse_does_not_request_dense(self, client: TestClient, sample_run) -> None:
        client.post("/api/v1/research/retrieve/sparse", json={"run_id": RUN, "query": "BERT"})
        client.post("/api/v1/research/retrieve/hybrid", json={"run_id": RUN, "query": "BERT"})
        assert sample_run == [(RUN, False), (RUN, True)]


class TestBuildRetrievalService:
    """_build_retrieval_service with mocked DB, storage and loader."""

    @pytest.fixture
    def deps(self, monkeypatch):
        import app.ingestion.corpus.run_corpus as run_corpus_mod
        import app.storage.relational.database as database
        import app.storage.repositories.paper_repository as paper_repo_mod
        import app.storage.supabase_storage as storage_mod
        from app.core.config import settings

        repo = MagicMock()
        repo.run_exists = AsyncMock(return_value=True)
        loader = MagicMock()
        loader.load = AsyncMock(return_value=make_sample_corpus())
        loader_cls = MagicMock(return_value=loader)

        monkeypatch.setattr(database, "get_db_pool", AsyncMock(return_value=object()))
        monkeypatch.setattr(paper_repo_mod, "PaperRepository", MagicMock(return_value=repo))
        monkeypatch.setattr(storage_mod, "SupabaseStorageClient", MagicMock())
        monkeypatch.setattr(run_corpus_mod, "RunCorpusLoader", loader_cls)
        monkeypatch.setattr(settings, "EMBEDDING_API_KEY", "", raising=False)
        return repo, loader, loader_cls

    async def test_loads_run_corpus(self, deps) -> None:
        _, loader, loader_cls = deps
        service = await _REAL_BUILD(RUN, with_dense=True)
        loader.load.assert_awaited_once_with(RUN)
        assert loader_cls.call_args.kwargs["embedding_repo"] is None  # no API key
        assert service.retrieve("GPT-3 limitations", top_k=1, method="sparse")[0].chunk_id == (
            "chunk-003-limits"
        )

    async def test_unknown_run_is_404(self, deps) -> None:
        from fastapi import HTTPException

        repo, loader, _ = deps
        repo.run_exists.return_value = False
        with pytest.raises(HTTPException) as exc:
            await _REAL_BUILD(RUN, with_dense=False)
        assert exc.value.status_code == 404
        loader.load.assert_not_awaited()

    async def test_unchunked_run_is_409(self, deps) -> None:
        from fastapi import HTTPException

        from app.ingestion.corpus.run_corpus import RunCorpus

        _, loader, _ = deps
        loader.load.return_value = RunCorpus([], [], run_id=RUN)
        with pytest.raises(HTTPException) as exc:
            await _REAL_BUILD(RUN, with_dense=False)
        assert exc.value.status_code == 409
        assert "/chunks" in exc.value.detail
