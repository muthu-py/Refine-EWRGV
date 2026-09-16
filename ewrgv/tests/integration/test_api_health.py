"""
tests/integration/test_api_health.py
--------------------------------------
Integration test for the FastAPI health endpoint.

Tests that:
- The application starts correctly.
- GET /api/v1/health returns HTTP 200.
- The response body matches the HealthResponse schema.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient):
        response = client.get("/api/v1/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client: TestClient):
        response = client.get("/api/v1/health")
        body = response.json()
        assert body["status"] == "ok"
        assert "app" in body
        assert "version" in body
        assert "timestamp" in body

    def test_health_app_name(self, client: TestClient):
        response = client.get("/api/v1/health")
        body = response.json()
        assert "EWRGV" in body["app"] or "Evidence" in body["app"]


class TestResearchEndpointsExist:
    """Smoke tests — just verify endpoints return non-404."""

    def test_submit_query_endpoint_exists(self, client: TestClient):
        response = client.post(
            "/api/v1/research/query",
            json={"query": "test query"},
        )
        assert response.status_code != 404

    def test_get_job_status_endpoint_exists(self, client: TestClient):
        response = client.get("/api/v1/research/some-job-id")
        assert response.status_code != 404
