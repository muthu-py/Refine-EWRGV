"""
tests/unit/test_retrieval_api.py
----------------------------------
Tests for the four retrieval API endpoints:

    POST /api/v1/research/retrieve/dense
    POST /api/v1/research/retrieve/sparse
    POST /api/v1/research/retrieve/knowledge-graph
    POST /api/v1/research/retrieve/hybrid

Uses FastAPI TestClient to make real HTTP calls against the app.
All tests run against the stub corpus (no external services).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


# ====================================================================
# Dense: POST /retrieve/dense
# ====================================================================


class TestRetrieveDense:
    """POST /api/v1/research/retrieve/dense."""

    URL = "/api/v1/research/retrieve/dense"

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"query": "Transformer attention", "top_k": 5})
        assert resp.status_code == 200

    def test_response_schema(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"query": "Transformer attention", "top_k": 3})
        body = resp.json()
        assert body["query"] == "Transformer attention"
        assert body["top_k"] == 3
        assert "total_results" in body
        assert isinstance(body["results"], list)
        assert len(body["results"]) > 0

    def test_result_items_have_required_fields(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"query": "neural network", "top_k": 3})
        for item in resp.json()["results"]:
            assert "chunk_id" in item
            assert "paper_id" in item
            assert "section" in item
            assert "text" in item
            assert "score" in item
            assert "rank" in item
            assert "retrieval_method" in item

    def test_retrieval_method_is_dense(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"query": "attention", "top_k": 3})
        for item in resp.json()["results"]:
            assert item["retrieval_method"] == "dense"

    def test_respects_top_k(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"query": "model", "top_k": 3})
        assert len(resp.json()["results"]) <= 3


# ====================================================================
# Sparse: POST /retrieve/sparse
# ====================================================================


class TestRetrieveSparse:
    """POST /api/v1/research/retrieve/sparse."""

    URL = "/api/v1/research/retrieve/sparse"

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"query": "Transformer attention", "top_k": 5})
        assert resp.status_code == 200

    def test_retrieval_method_is_sparse(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"query": "language model", "top_k": 3})
        for item in resp.json()["results"]:
            assert item["retrieval_method"] == "sparse"

    def test_respects_top_k(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"query": "Transformer", "top_k": 2})
        assert len(resp.json()["results"]) <= 2


# ====================================================================
# Knowledge Graph: POST /retrieve/knowledge-graph
# ====================================================================


class TestRetrieveKnowledgeGraph:
    """POST /api/v1/research/retrieve/knowledge-graph."""

    URL = "/api/v1/research/retrieve/knowledge-graph"

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"query": "Transformer", "top_k": 5})
        assert resp.status_code == 200

    def test_retrieval_method_is_knowledge_graph(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"query": "BERT", "top_k": 3})
        for item in resp.json()["results"]:
            assert item["retrieval_method"] == "knowledge_graph"

    def test_metadata_includes_entities(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"query": "Transformer", "top_k": 3})
        results = resp.json()["results"]
        has_entity_info = any(
            r["metadata"].get("matched_entities") or r["metadata"].get("relations")
            for r in results
        )
        assert has_entity_info

    def test_respects_top_k(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"query": "Transformer", "top_k": 2})
        assert len(resp.json()["results"]) <= 2


# ====================================================================
# Hybrid: POST /retrieve/hybrid
# ====================================================================


class TestRetrieveHybrid:
    """POST /api/v1/research/retrieve/hybrid."""

    URL = "/api/v1/research/retrieve/hybrid"

    def test_returns_200(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"query": "Transformer attention", "top_k": 5})
        assert resp.status_code == 200

    def test_retrieval_method_is_hybrid(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"query": "language model", "top_k": 3})
        for item in resp.json()["results"]:
            assert item["retrieval_method"] == "hybrid"

    def test_no_duplicate_chunks(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"query": "Transformer model", "top_k": 20})
        ids = [r["chunk_id"] for r in resp.json()["results"]]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs in hybrid results"

    def test_provenance_metadata(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"query": "Transformer", "top_k": 5})
        for item in resp.json()["results"]:
            assert "contributing_methods" in item["metadata"]

    def test_respects_top_k(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"query": "Transformer", "top_k": 3})
        assert len(resp.json()["results"]) <= 3


# ====================================================================
# Default top_k
# ====================================================================


class TestRetrieveDefaults:
    """Default parameter behavior across all endpoints."""

    def test_default_top_k_is_10(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/research/retrieve/hybrid",
            json={"query": "language model"},
        )
        assert resp.json()["top_k"] == 10


# ====================================================================
# Validation — shared across all endpoints
# ====================================================================


class TestRetrieveValidation:
    """Validation / error handling (tested against /retrieve/dense as representative)."""

    URL = "/api/v1/research/retrieve/dense"

    def test_empty_query_returns_400(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"query": ""})
        assert resp.status_code == 400

    def test_whitespace_query_returns_400(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"query": "   "})
        assert resp.status_code == 400

    def test_missing_query_returns_422(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"top_k": 5})
        assert resp.status_code == 422

    def test_top_k_zero_returns_400(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"query": "test", "top_k": 0})
        assert resp.status_code == 400

    def test_top_k_negative_returns_400(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"query": "test", "top_k": -5})
        assert resp.status_code == 400

    def test_top_k_too_large_returns_400(self, client: TestClient) -> None:
        resp = client.post(self.URL, json={"query": "test", "top_k": 999})
        assert resp.status_code == 400

    def test_validation_on_sparse(self, client: TestClient) -> None:
        """Validation works on /sparse too."""
        resp = client.post("/api/v1/research/retrieve/sparse", json={"query": ""})
        assert resp.status_code == 400

    def test_validation_on_kg(self, client: TestClient) -> None:
        """Validation works on /knowledge-graph too."""
        resp = client.post("/api/v1/research/retrieve/knowledge-graph", json={"query": ""})
        assert resp.status_code == 400

    def test_validation_on_hybrid(self, client: TestClient) -> None:
        """Validation works on /hybrid too."""
        resp = client.post("/api/v1/research/retrieve/hybrid", json={"query": ""})
        assert resp.status_code == 400


# ====================================================================
# Old /retrieve endpoint is removed
# ====================================================================


class TestOldEndpointRemoved:
    """The generic /retrieve endpoint should no longer exist."""

    def test_old_retrieve_returns_404_or_405(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/research/retrieve",
            json={"query": "test", "method": "dense"},
        )
        # Should not be 200 — it's been removed
        assert resp.status_code in (404, 405, 307)


# ====================================================================
# Determinism
# ====================================================================


class TestRetrieveDeterminism:
    """Results are deterministic for the same input."""

    def test_dense_deterministic(self, client: TestClient) -> None:
        payload = {"query": "BERT pre-training", "top_k": 5}
        r1 = client.post("/api/v1/research/retrieve/dense", json=payload).json()
        r2 = client.post("/api/v1/research/retrieve/dense", json=payload).json()
        assert [r["chunk_id"] for r in r1["results"]] == [r["chunk_id"] for r in r2["results"]]

    def test_hybrid_deterministic(self, client: TestClient) -> None:
        payload = {"query": "BERT pre-training", "top_k": 5}
        r1 = client.post("/api/v1/research/retrieve/hybrid", json=payload).json()
        r2 = client.post("/api/v1/research/retrieve/hybrid", json=payload).json()
        assert [r["chunk_id"] for r in r1["results"]] == [r["chunk_id"] for r in r2["results"]]


# ====================================================================
# Ranks
# ====================================================================


class TestRetrieveRanks:
    """Rank ordering validation."""

    def test_ranks_are_sequential_dense(self, client: TestClient) -> None:
        resp = client.post("/api/v1/research/retrieve/dense", json={"query": "attention", "top_k": 5})
        for i, r in enumerate(resp.json()["results"]):
            assert r["rank"] == i + 1

    def test_ranks_are_sequential_hybrid(self, client: TestClient) -> None:
        resp = client.post("/api/v1/research/retrieve/hybrid", json={"query": "attention", "top_k": 5})
        for i, r in enumerate(resp.json()["results"]):
            assert r["rank"] == i + 1
