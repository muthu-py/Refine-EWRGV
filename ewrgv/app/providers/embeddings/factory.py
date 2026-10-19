"""
app/providers/embeddings/factory.py
-------------------------------------
Factory for instantiating embedding providers based on configuration.

Parallels the LLM provider factory (app/providers/llm/factory.py).
"""

from __future__ import annotations

from app.core.config import settings
from app.core.logging import get_logger
from app.providers.embeddings.openai_embeddings import OpenAIEmbeddingProvider

logger = get_logger(__name__)

# OpenAI-compatible embedding endpoints, keyed by EMBEDDING_PROVIDER.
_DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai",
    "openrouter": "https://openrouter.ai/api/v1",
}


def create_embedding_provider(
    provider_name: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    dimension: int | None = None,
    timeout: int | None = None,
    max_retries: int | None = None,
    retry_delay: float | None = None,
) -> OpenAIEmbeddingProvider:
    """
    Factory to construct an EmbeddingProvider instance.

    If parameters are omitted, values from application Settings (.env) are used.

    Supports OpenAI-compatible endpoints: 'openai', 'google'/'gemini', 'openrouter'.

    Returns
    -------
    OpenAIEmbeddingProvider:
        Configured embedding provider instance.

    Raises
    ------
    ValueError:
        If an unsupported provider name is specified.
    """
    name = (provider_name or settings.EMBEDDING_PROVIDER or "openai").lower().strip()

    key = api_key or settings.EMBEDDING_API_KEY
    mdl = model or settings.EMBEDDING_MODEL
    dim = dimension or settings.EMBEDDING_DIMENSION
    tout = timeout or settings.EMBEDDING_TIMEOUT
    retries = max_retries if max_retries is not None else settings.EMBEDDING_RETRY_COUNT
    delay = retry_delay if retry_delay is not None else settings.EMBEDDING_RETRY_DELAY

    if name in _DEFAULT_BASE_URLS:
        base_url = settings.EMBEDDING_BASE_URL or _DEFAULT_BASE_URLS[name]
        logger.info(
            "Creating %s embedding provider (model=%s, dim=%d)",
            name,
            mdl,
            dim,
        )
        return OpenAIEmbeddingProvider(
            api_key=key,
            model=mdl,
            dimension=dim,
            timeout=tout,
            max_retries=retries,
            retry_delay=delay,
            base_url=base_url,
        )

    raise ValueError(
        f"Unsupported embedding provider: '{name}'. "
        f"Supported providers: {', '.join(sorted(_DEFAULT_BASE_URLS))}."
    )
