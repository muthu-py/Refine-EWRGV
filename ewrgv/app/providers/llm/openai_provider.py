"""
app/providers/llm/openai_provider.py
--------------------------------------
OpenAI LLM provider implementation.

Implements the LLMProvider interface using the openai Python SDK.

Status: SKELETON – correct interface, not yet implemented.
"""

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


class OpenAILLMProvider:
    """
    LLMProvider implementation backed by the OpenAI API.

    Satisfies the LLMProvider protocol (app.domain.interfaces).

    Status: SKELETON – not yet implemented.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        # self._client = openai.OpenAI(api_key=api_key)  ← add when implemented

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Return a text completion. NOT YET IMPLEMENTED."""
        logger.warning("OpenAILLMProvider.complete is not yet implemented.")
        return ""

    def complete_structured(self, prompt: str, schema: dict, **kwargs: Any) -> dict:
        """Return a structured JSON completion. NOT YET IMPLEMENTED."""
        logger.warning("OpenAILLMProvider.complete_structured is not yet implemented.")
        return {}
