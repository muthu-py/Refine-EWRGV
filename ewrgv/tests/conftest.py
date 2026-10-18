"""
tests/conftest.py
------------------
Suite-wide test configuration.

LLM credential isolation
------------------------
Tests must never use real LLM credentials, even when a developer has a
key in ``.env`` or the environment.  Without this, any test that reaches a
code path which builds a provider from settings (e.g. POST
/api/v1/research/query via ``_get_qu_service``) would make a live, billed
network call.

The autouse fixture blanks the LLM key settings for every test, so such
paths fall back to their documented no-key behaviour.  It only patches the
in-memory settings object for the duration of each test; it does not touch
``.env`` or the real provider configuration.  A test that needs a key value
(e.g. to check redaction) can still monkeypatch one explicitly.
"""

from __future__ import annotations

import pytest

from app.core.config import settings

_LLM_CREDENTIAL_SETTINGS = ("LLM_API_KEY", "OPENROUTER_API_KEY")


@pytest.fixture(autouse=True)
def _no_real_llm_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _LLM_CREDENTIAL_SETTINGS:
        monkeypatch.setattr(settings, name, "")
