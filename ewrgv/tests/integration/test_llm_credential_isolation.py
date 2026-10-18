"""
tests/integration/test_llm_credential_isolation.py
----------------------------------------------------
Regression tests: the test suite never uses real LLM credentials.

Before tests/conftest.py blanked the LLM key settings, POST
/api/v1/research/query built a real provider whenever a key was present in
``.env`` or the environment, so a plain ``pytest`` run could make a live,
billed call to the LLM provider.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.providers.llm.factory as llm_factory
from app.core.config import settings
from app.main import app


def test_llm_credentials_blank_inside_tests():
    assert settings.LLM_API_KEY == ""
    assert settings.OPENROUTER_API_KEY == ""


def test_query_endpoint_stays_offline(monkeypatch: pytest.MonkeyPatch):
    def _fail(*args, **kwargs):
        raise AssertionError("tests must not build a real LLM provider")

    monkeypatch.setattr(llm_factory, "create_llm_provider", _fail)

    response = TestClient(app).post("/api/v1/research/query", json={"query": "test query"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "accepted"  # documented no-key stub mode
    assert body["understanding"] is None
    assert body["expanded_queries"] == []


def test_explicit_override_still_possible(monkeypatch: pytest.MonkeyPatch):
    """Individual tests can still set a placeholder value when they need one."""
    monkeypatch.setattr(settings, "LLM_API_KEY", "placeholder-not-a-real-key")
    assert settings.LLM_API_KEY == "placeholder-not-a-real-key"
