"""
tests/fixtures/fake_llm.py
----------------------------
Strict, deterministic offline LLMProvider for pipeline tests.

Implements exactly the LLMProvider protocol — calling any other method
raises AttributeError, unlike an unspecced MagicMock.  No API key, no network.
"""

from __future__ import annotations


class ScriptedFakeLLM:
    """
    Deterministic offline LLMProvider.

    Answers ``complete_structured`` by recognising which schema it was given
    (query understanding vs gap detection) and records every call.
    """

    def __init__(self, understanding: dict, detection: dict) -> None:
        self._understanding = understanding
        self._detection = detection
        self.calls: list[tuple[str, str]] = []  # (kind, prompt)

    def complete(self, prompt: str, **kwargs) -> str:
        raise AssertionError("pipeline Stages 1-7 should only use complete_structured")

    def complete_structured(self, prompt: str, schema: dict, **kwargs) -> dict:
        properties = schema.get("properties", {})
        if "candidate_gaps" in properties:
            self.calls.append(("detection", prompt))
            return self._detection
        if "expanded_queries" in properties:
            self.calls.append(("understanding", prompt))
            return self._understanding
        raise AssertionError(f"unexpected schema: {sorted(properties)}")
