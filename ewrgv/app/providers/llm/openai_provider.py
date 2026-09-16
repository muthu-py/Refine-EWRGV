"""
app/providers/llm/openai_provider.py
--------------------------------------
OpenAI LLM provider implementation.

Implements the LLMProvider interface (app.domain.interfaces.LLMProvider)
using the openai Python SDK.

Design notes
------------
* The provider is stateless beyond its configuration parameters.
* complete_structured() uses OpenAI's JSON-mode (response_format=json_object)
  so that the model returns valid JSON conforming to the requested schema.
  Function-calling / structured-outputs mode is NOT used here to keep the
  implementation SDK-version agnostic and the interface clean.
* All SDK-level errors are caught and re-raised as ProviderError so that
  callers depend only on the EWRGV exception hierarchy.
* Timeout is applied at the SDK call level.
"""

from __future__ import annotations

import json
from typing import Any

from app.core.exceptions import ProviderError
from app.core.logging import get_logger

logger = get_logger(__name__)


class OpenAILLMProvider:
    """
    LLMProvider implementation backed by the OpenAI Chat Completions API.

    Satisfies the LLMProvider protocol (app.domain.interfaces.LLMProvider).

    Parameters
    ----------
    api_key:
        OpenAI API key.
    model:
        Chat completion model identifier (e.g. "gpt-4o", "gpt-4o-mini").
    temperature:
        Sampling temperature.  0.0 recommended for deterministic structured
        outputs.
    max_tokens:
        Maximum number of tokens in the completion.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str | None = None,
        default_headers: dict[str, str] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 30,
    ) -> None:
        if not api_key:
            raise ValueError(
                f"{type(self).__name__} requires a non-empty api_key. "
                "Set LLM_API_KEY in your environment or .env file."
            )
        self._api_key = api_key
        self._model = model
        self._base_url = base_url
        self._default_headers = default_headers
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._client = self._build_client()

    # ------------------------------------------------------------------ #
    # Public interface (satisfies LLMProvider protocol)
    # ------------------------------------------------------------------ #

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """
        Return a free-text completion for the given prompt.

        Parameters
        ----------
        prompt:
            The full prompt to send to the model.
        **kwargs:
            Optional overrides: temperature, max_tokens.

        Returns
        -------
        str
            The model's text completion.

        Raises
        ------
        ProviderError
            On any OpenAI API error, network failure, or timeout.
        """
        logger.debug(
            f"{type(self).__name__}.complete called",
            extra={"model": self._model, "prompt_len": len(prompt)},
        )
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self._temperature),
                max_tokens=kwargs.get("max_tokens", self._max_tokens),
                timeout=self._timeout,
            )
            content = response.choices[0].message.content or ""
            logger.debug(
                f"{type(self).__name__}.complete succeeded",
                extra={"model": self._model, "response_len": len(content)},
            )
            return content
        except Exception as exc:
            raise ProviderError(
                f"{type(self).__name__} complete() failed: {exc}",
                detail=str(exc),
            ) from exc

    def complete_structured(
        self, prompt: str, schema: dict, **kwargs: Any
    ) -> dict:
        """
        Return a JSON-structured completion conforming to ``schema``.

        The model is instructed via a system message to return valid JSON,
        and OpenAI's ``response_format={"type": "json_object"}`` is enabled
        so that the response is guaranteed to be parseable JSON.

        The caller is still responsible for validating the returned dict
        against the expected schema (e.g. via Pydantic).

        Parameters
        ----------
        prompt:
            The user-level prompt describing the task.
        schema:
            A JSON Schema dict passed to the model as context.  The provider
            includes it in a system message so the model knows the required
            structure.
        **kwargs:
            Optional overrides: temperature, max_tokens.

        Returns
        -------
        dict
            Parsed JSON dict from the model's completion.

        Raises
        ------
        ProviderError
            On API error, network failure, timeout, or unparseable JSON.
        """
        logger.debug(
            f"{type(self).__name__}.complete_structured called",
            extra={"model": self._model},
        )
        system_message = (
            "You are a structured data extraction assistant. "
            "Always respond with a single valid JSON object that conforms "
            "to the schema provided by the user. "
            "Do not add markdown fences, explanatory text, or any content "
            "outside the JSON object."
        )
        schema_context = (
            f"\n\nRequired JSON schema:\n{json.dumps(schema, indent=2)}"
        )
        full_prompt = prompt + schema_context

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": full_prompt},
                ],
                temperature=kwargs.get("temperature", self._temperature),
                max_tokens=kwargs.get("max_tokens", self._max_tokens),
                response_format={"type": "json_object"},
                timeout=self._timeout,
            )
            raw = response.choices[0].message.content or "{}"
            logger.debug(
                f"{type(self).__name__}.complete_structured: raw response received",
                extra={"model": self._model, "response_len": len(raw)},
            )
            raw_clean = raw.strip()
            if raw_clean.startswith("```"):
                lines = raw_clean.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                raw_clean = "\n".join(lines).strip()
            return json.loads(raw_clean)
        except json.JSONDecodeError as exc:
            raise ProviderError(
                f"{type(self).__name__} complete_structured() returned non-JSON content.",
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise ProviderError(
                f"{type(self).__name__} complete_structured() failed: {exc}",
                detail=str(exc),
            ) from exc

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_client(self) -> Any:
        """
        Lazily import and instantiate the OpenAI SDK client.

        Importing here (rather than at module level) keeps the rest of the
        application importable even when the ``openai`` package is not
        installed — as long as this provider is never instantiated.
        """
        try:
            import openai  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                f"The 'openai' package is required to use {type(self).__name__}. "
                "Install it with: pip install openai>=1.30.0"
            ) from exc
        client_kwargs: dict[str, Any] = {"api_key": self._api_key}
        if self._base_url:
            client_kwargs["base_url"] = self._base_url
        if self._default_headers:
            client_kwargs["default_headers"] = self._default_headers
        return openai.OpenAI(**client_kwargs)
