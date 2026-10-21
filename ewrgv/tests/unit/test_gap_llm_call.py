"""
tests/unit/test_gap_llm_call.py
---------------------------------
Unit tests for the shared gap-detection LLM call helper (llm_call.py).

Covers M3 (provider cause preserved in message, detail and logs) and
secret redaction, plus the shared error contract used by GapDetector and
GapRepresentationBuilder (L5).
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from app.core.exceptions import GapDetectionError, ProviderError
from app.gap_detection.llm_call import call_structured_llm, redact_secrets


def _llm(return_value=None, raises=None) -> MagicMock:
    llm = MagicMock()
    if raises is not None:
        llm.complete_structured.side_effect = raises
    else:
        llm.complete_structured.return_value = return_value
    return llm


def _call(llm: MagicMock) -> dict:
    return call_structured_llm(
        llm, "prompt", {"type": "object"}, operation="gap detection",
        log_extra={"query_id": "q-1"},
    )


class TestCallStructuredLLM:
    def test_returns_dict(self):
        assert _call(_llm({"a": 1})) == {"a": 1}

    def test_passes_prompt_and_schema(self):
        llm = _llm({})
        _call(llm)
        llm.complete_structured.assert_called_once_with("prompt", {"type": "object"})

    def test_non_dict_raises(self):
        with pytest.raises(GapDetectionError) as exc_info:
            _call(_llm([1, 2]))
        assert exc_info.value.detail == "Type received: list"

    def test_rate_limit_cause_in_message_detail_and_log(self, caplog):
        err = ProviderError(
            "OpenRouterLLMProvider complete_structured() failed: Error code: 429 - rate limit",
            detail="Error code: 429 - {'error': {'message': 'Rate limit exceeded'}}",
        )
        with caplog.at_level(logging.WARNING, logger="app.gap_detection.llm_call"):
            with pytest.raises(GapDetectionError) as exc_info:
                _call(_llm(raises=err))

        exc = exc_info.value
        assert "429" in exc.message and "429" in str(exc)
        assert "gap detection" in exc.message
        assert "Rate limit exceeded" in exc.detail
        assert isinstance(exc.__cause__, ProviderError)
        record = caplog.records[-1]
        assert "429" in record.getMessage()
        assert record.query_id == "q-1"
        assert "Rate limit exceeded" in record.error_detail

    def test_detail_falls_back_to_message(self):
        with pytest.raises(GapDetectionError) as exc_info:
            _call(_llm(raises=ProviderError("timeout after 30s")))
        assert exc_info.value.detail == "timeout after 30s"

    def test_unexpected_exception_includes_type_and_cause(self):
        with pytest.raises(GapDetectionError) as exc_info:
            _call(_llm(raises=ConnectionError("connection reset")))
        assert "ConnectionError: connection reset" in exc_info.value.message

    def test_long_cause_truncated_in_message(self):
        with pytest.raises(GapDetectionError) as exc_info:
            _call(_llm(raises=ProviderError("x" * 5000)))
        assert len(exc_info.value.message) < 400
        assert len(exc_info.value.detail) == 5000

    def test_secrets_not_exposed_in_error_or_logs(self, caplog):
        secret = "sk-proj-AbCdEf1234567890XYZ"
        err = ProviderError(
            f"Error code: 401 - Incorrect API key provided: {secret}",
            detail=f"headers: Authorization: Bearer {secret}; api_key={secret}",
        )
        with caplog.at_level(logging.WARNING, logger="app.gap_detection.llm_call"):
            with pytest.raises(GapDetectionError) as exc_info:
                _call(_llm(raises=err))

        exc = exc_info.value
        assert "401" in exc.message  # the useful part survives
        for surfaced in (exc.message, exc.detail, caplog.text,
                         caplog.records[-1].error_detail):
            assert secret not in surfaced
            assert "AbCdEf1234567890" not in surfaced


class TestRedactSecrets:
    @pytest.mark.parametrize(
        "text",
        [
            "key sk-abcdef123456 rejected",
            "key sk-or-v1-0123456789abcdef rejected",
            "key xai-ABCDEF123456 rejected",
            "Authorization: Bearer eyJhbGciOi.payload.sig",
            "url?api_key=supersecretvalue&x=1",
            '{"token": "supersecretvalue"}',
            "password=hunter2hunter2",
        ],
    )
    def test_patterns_redacted(self, text):
        redacted = redact_secrets(text)
        assert "[REDACTED]" in redacted
        for fragment in ("abcdef123456", "0123456789abcdef", "ABCDEF123456",
                         "eyJhbGciOi", "supersecretvalue", "hunter2hunter2"):
            assert fragment not in redacted

    def test_configured_secret_values_redacted(self, monkeypatch):
        from app.core.config import settings

        monkeypatch.setattr(settings, "LLM_API_KEY", "plainKeyWithoutPrefix42")
        assert "plainKeyWithoutPrefix42" not in redact_secrets(
            "upstream echoed plainKeyWithoutPrefix42 back"
        )

    def test_ordinary_text_untouched(self):
        text = "Error code: 429 - Rate limit exceeded for model openai/gpt-4o-mini"
        assert redact_secrets(text) == text

    def test_empty(self):
        assert redact_secrets(None) == ""
        assert redact_secrets("") == ""
