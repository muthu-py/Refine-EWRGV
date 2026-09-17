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

    Currently supports: 'openai'.

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

    if name == "openai":
        logger.info(
            "Creating OpenAI embedding provider (model=%s, dim=%d)",
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
        )

    raise ValueError(
        f"Unsupported embedding provider: '{name}'. "
        "Supported providers: 'openai'."
    )
