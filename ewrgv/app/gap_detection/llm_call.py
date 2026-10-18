"""
app/gap_detection/llm_call.py
-------------------------------
Shared structured-LLM call for the gap detection package.

Used by GapDetector and GapRepresentationBuilder so that both handle
provider failures identically:

* Any failure (ProviderError, unexpected exception, non-dict response)
  is raised as GapDetectionError.
* The underlying cause (e.g. "Error code: 429 - rate limit exceeded") is
  kept in the error message, the error detail, and the warning log, so it
  reaches PipelineResult.error and the logs.
* Secrets are redacted from everything that is surfaced: API-key-like
  tokens, bearer tokens, key/secret/password assignments, and the literal
  values of configured credentials.
"""

from __future__ import annotations

import re

from app.core.exceptions import GapDetectionError, ProviderError
from app.core.logging import get_logger
from app.domain.interfaces import LLMProvider

logger = get_logger(__name__)

# Maximum length of the cause appended to the error message.
_MAX_CAUSE_CHARS = 300

_REDACTED = "[REDACTED]"
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # OpenAI / OpenRouter style keys: sk-..., sk-proj-..., sk-or-v1-...
    (re.compile(r"\bsk-[A-Za-z0-9_\-]{6,}"), _REDACTED),
    # xAI keys
    (re.compile(r"\bxai-[A-Za-z0-9_\-]{6,}"), _REDACTED),
    # Authorization: Bearer <token>
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/\-]+=*"), f"Bearer {_REDACTED}"),
    # api_key=..., "token": "...", password: ...
    (
        re.compile(
            r"(?i)\b(api[_-]?key|access[_-]?token|token|secret|password|authorization)"
            r"(\s*[\"']?\s*[:=]\s*[\"']?)([^\s\"',;&]+)"
        ),
        rf"\1\2{_REDACTED}",
    ),
)

# Settings fields holding credentials; their literal values are redacted too.
_CREDENTIAL_SETTINGS = (
    "LLM_API_KEY",
    "OPENROUTER_API_KEY",
    "EMBEDDING_API_KEY",
    "SEMANTIC_SCHOLAR_API_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
)


def _configured_secrets() -> list[str]:
    from app.core.config import settings  # noqa: PLC0415

    values = (getattr(settings, name, "") or "" for name in _CREDENTIAL_SETTINGS)
    # Very short values would cause false-positive redaction of ordinary words.
    return [v for v in values if len(v) >= 8]


def redact_secrets(text: str | None) -> str:
    """Return ``text`` with credentials and credential-like tokens redacted."""
    if not text:
        return ""
    redacted = text
    for secret in _configured_secrets():
        redacted = redacted.replace(secret, _REDACTED)
    for pattern, replacement in _SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted


def _truncate(text: str, limit: int = _MAX_CAUSE_CHARS) -> str:
    return text if len(text) <= limit else text[:limit].rstrip() + " …"


def call_structured_llm(
    llm: LLMProvider,
    prompt: str,
    schema: dict,
    *,
    operation: str,
    log_extra: dict | None = None,
) -> dict:
    """
    Call ``llm.complete_structured`` and return its dict response.

    Parameters
    ----------
    operation:
        Short human-readable name used in messages, e.g. "gap detection".
    log_extra:
        Extra structured log fields (e.g. ``{"query_id": ...}``).

    Raises
    ------
    GapDetectionError
        On provider failure, unexpected exception, or a non-dict response.
        The redacted cause is included in ``message`` and ``detail``.
    """
    extra = dict(log_extra or {})
    try:
        result = llm.complete_structured(prompt, schema)
    except ProviderError as exc:
        cause = redact_secrets(exc.message)
        detail = redact_secrets(exc.detail) if exc.detail else cause
        logger.warning(
            "LLM provider failure during %s: %s",
            operation,
            _truncate(cause),
            extra={**extra, "error_detail": _truncate(detail, 1000)},
        )
        raise GapDetectionError(
            f"LLM provider failed during {operation}: {_truncate(cause)}",
            detail=detail,
        ) from exc
    except Exception as exc:
        cause = redact_secrets(f"{type(exc).__name__}: {exc}")
        logger.warning(
            "Unexpected error from LLM during %s: %s",
            operation,
            _truncate(cause),
            extra=extra,
        )
        raise GapDetectionError(
            f"Unexpected error during {operation} LLM call: {_truncate(cause)}",
            detail=cause,
        ) from exc

    if not isinstance(result, dict):
        raise GapDetectionError(
            f"LLM provider returned a non-dict response during {operation}.",
            detail=f"Type received: {type(result).__name__}",
        )
    return result
