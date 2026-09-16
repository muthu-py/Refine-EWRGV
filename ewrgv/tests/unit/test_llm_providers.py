"""
tests/unit/test_llm_providers.py
----------------------------------
Unit tests for LLM providers (OpenAI, OpenRouter) and factory.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.core.exceptions import ProviderError
from app.providers.llm.factory import create_llm_provider
from app.providers.llm.openai_provider import OpenAILLMProvider
from app.providers.llm.openrouter_provider import OpenRouterLLMProvider


def test_openrouter_provider_initialization():
    """Verify OpenRouterLLMProvider initializes with OpenRouter defaults."""
    with patch("openai.OpenAI") as mock_openai:
        provider = OpenRouterLLMProvider(api_key="sk-or-v1-test")
        assert provider._model == "openai/gpt-4o-mini"
        assert provider._base_url == "https://openrouter.ai/api/v1"
        mock_openai.assert_called_once_with(
            api_key="sk-or-v1-test",
            base_url="https://openrouter.ai/api/v1",
            default_headers={"X-Title": "EWRGV Research Gap Validation"},
        )


def test_openrouter_provider_custom_model_and_site():
    """Verify OpenRouterLLMProvider allows overriding model and setting site_url."""
    with patch("openai.OpenAI") as mock_openai:
        provider = OpenRouterLLMProvider(
            api_key="sk-or-v1-test",
            model="anthropic/claude-3.5-sonnet",
            site_url="https://example.com",
            app_name="TestApp",
        )
        assert provider._model == "anthropic/claude-3.5-sonnet"
        assert provider._base_url == "https://openrouter.ai/api/v1"
        mock_openai.assert_called_once_with(
            api_key="sk-or-v1-test",
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "X-Title": "TestApp",
                "HTTP-Referer": "https://example.com",
            },
        )


def test_openai_provider_with_base_url():
    """Verify OpenAILLMProvider passes base_url to OpenAI client."""
    with patch("openai.OpenAI") as mock_openai:
        provider = OpenAILLMProvider(
            api_key="sk-test",
            base_url="https://proxy.example.com/v1",
        )
        assert provider._base_url == "https://proxy.example.com/v1"
        mock_openai.assert_called_once_with(
            api_key="sk-test",
            base_url="https://proxy.example.com/v1",
        )


def test_create_llm_provider_factory_openrouter():
    """Verify factory builds OpenRouterLLMProvider for 'openrouter'."""
    with patch("openai.OpenAI"):
        provider = create_llm_provider(
            provider_name="openrouter",
            api_key="sk-or-v1-factory",
        )
        assert isinstance(provider, OpenRouterLLMProvider)
        assert provider._base_url == "https://openrouter.ai/api/v1"


def test_create_llm_provider_factory_openai():
    """Verify factory builds OpenAILLMProvider for 'openai'."""
    with patch("openai.OpenAI"):
        provider = create_llm_provider(
            provider_name="openai",
            api_key="sk-factory",
            model="gpt-4o",
        )
        assert isinstance(provider, OpenAILLMProvider)
        assert not isinstance(provider, OpenRouterLLMProvider)
        assert provider._model == "gpt-4o"


def test_create_llm_provider_factory_unsupported():
    """Verify factory raises ValueError for unknown provider."""
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        create_llm_provider(
            provider_name="unsupported_provider",
            api_key="some-key",
        )


def test_complete_structured_strips_markdown_fences():
    """Verify complete_structured strips ```json fences if returned by LLM."""
    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = '```json\n{"domain": "AI", "concepts": ["GNN"]}\n```'
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        provider = OpenRouterLLMProvider(api_key="sk-or-v1-test")
        result = provider.complete_structured(prompt="Analyze this", schema={})

        assert result == {"domain": "AI", "concepts": ["GNN"]}
