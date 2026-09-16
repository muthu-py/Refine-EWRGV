"""
app/providers/llm/openrouter_provider.py
------------------------------------------
OpenRouter LLM provider implementation.

Implements the LLMProvider interface (app.domain.interfaces.LLMProvider)
using the openai Python SDK pointing to OpenRouter's API endpoint (https://openrouter.ai/api/v1).

Design notes
------------
* OpenRouter provides a unified OpenAI-compatible REST API for hundreds of LLMs
  (e.g. Claude 3.5 Sonnet, GPT-4o-mini, Gemini 2.0 Flash, Llama 3.3 70B, DeepSeek, etc.).
* Subclasses OpenAILLMProvider with defaults tailored for OpenRouter:
  - base_url: https://openrouter.ai/api/v1
  - default model: openai/gpt-4o-mini
  - default headers: HTTP-Referer and X-Title (recommended by OpenRouter)
* Automatically inherits JSON-mode completion, markdown-fence stripping, and error handling.
"""

from __future__ import annotations

from typing import Any

from app.providers.llm.openai_provider import OpenAILLMProvider


class OpenRouterLLMProvider(OpenAILLMProvider):
    """
    LLMProvider implementation backed by the OpenRouter Chat Completions API.

    Satisfies the LLMProvider protocol (app.domain.interfaces.LLMProvider).

    Parameters
    ----------
    api_key:
        OpenRouter API key (typically starts with 'sk-or-v1-').
    model:
        Model identifier on OpenRouter (e.g. "openai/gpt-4o-mini",
        "anthropic/claude-3.5-sonnet", "google/gemini-2.0-flash-001").
    base_url:
        API endpoint (defaults to "https://openrouter.ai/api/v1").
    site_url:
        App URL sent via HTTP-Referer header for OpenRouter analytics/ranking.
    app_name:
        App name sent via X-Title header.
    temperature:
        Sampling temperature. 0.0 recommended for deterministic structured outputs.
    max_tokens:
        Maximum number of tokens in completion.
    timeout:
        HTTP request timeout in seconds.
    """

    DEFAULT_BASE_URL: str = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL: str = "openai/gpt-4o-mini"
    DEFAULT_APP_NAME: str = "EWRGV Research Gap Validation"

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        site_url: str | None = None,
        app_name: str = DEFAULT_APP_NAME,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 30,
        **extra: Any,
    ) -> None:
        headers: dict[str, str] = {
            "X-Title": app_name,
        }
        if site_url:
            headers["HTTP-Referer"] = site_url

        super().__init__(
            api_key=api_key,
            model=model or self.DEFAULT_MODEL,
            base_url=base_url or self.DEFAULT_BASE_URL,
            default_headers=headers,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
