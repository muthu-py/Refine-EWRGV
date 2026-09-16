"""
app/providers/llm/factory.py
------------------------------
Factory for instantiating LLM providers based on configuration.
"""

from __future__ import annotations

from typing import Any

from app.core.config import settings
from app.domain.interfaces import LLMProvider
from app.providers.llm.openai_provider import OpenAILLMProvider
from app.providers.llm.openrouter_provider import OpenRouterLLMProvider


def create_llm_provider(
    provider_name: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
    **extra: Any,
) -> LLMProvider:
    """
    Factory to construct an LLMProvider instance.

    If parameters are omitted, values from application Settings (.env) are used.
    Supports 'openrouter', 'openai', 'grok', and 'xai'.
    """
    name = (provider_name or settings.LLM_PROVIDER or "openrouter").lower().strip()

    # Determine API key
    key = api_key or settings.LLM_API_KEY
    if not key and name in ("openrouter", "open_router"):
        key = settings.OPENROUTER_API_KEY

    temp = settings.LLM_TEMPERATURE if temperature is None else temperature
    tokens = settings.LLM_MAX_TOKENS if max_tokens is None else max_tokens
    tout = settings.LLM_TIMEOUT if timeout is None else timeout
    url = base_url or settings.LLM_BASE_URL

    if name in ("openrouter", "open_router"):
        # Resolve OpenRouter model (default to settings.LLM_MODEL or OpenRouterLLMProvider default)
        router_model = model or settings.LLM_MODEL
        if not router_model or router_model == "gpt-4o":
            router_model = OpenRouterLLMProvider.DEFAULT_MODEL
        return OpenRouterLLMProvider(
            api_key=key,
            model=router_model,
            base_url=url or OpenRouterLLMProvider.DEFAULT_BASE_URL,
            temperature=temp,
            max_tokens=tokens,
            timeout=tout,
            **extra,
        )

    if name in ("grok", "xai"):
        grok_model = model or settings.LLM_MODEL
        if not grok_model or grok_model == "gpt-4o":
            grok_model = "grok-2-latest"
        return OpenAILLMProvider(
            api_key=key,
            model=grok_model,
            base_url=url or "https://api.x.ai/v1",
            temperature=temp,
            max_tokens=tokens,
            timeout=tout,
        )

    if name == "openai":
        openai_model = model or settings.LLM_MODEL or "gpt-4o"
        return OpenAILLMProvider(
            api_key=key,
            model=openai_model,
            base_url=url,
            temperature=temp,
            max_tokens=tokens,
            timeout=tout,
        )

    raise ValueError(
        f"Unsupported LLM provider: '{name}'. "
        "Supported providers are 'openrouter', 'openai', and 'grok'."
    )
