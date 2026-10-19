"""
app/core/config.py
------------------
Centralised application configuration loaded from environment variables / .env file.

All settings are read once at startup via Pydantic BaseSettings, making every
configuration point explicit and type-checked.  No secrets are hard-coded here;
use the .env.example file as the reference template.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application-wide settings.

    Priority order (highest → lowest):
        1. OS environment variables
        2. .env file
        3. Field defaults defined here
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Application
    # ------------------------------------------------------------------ #
    APP_NAME: str = "EWRGV – Evidence-Weighted Research Gap Validation"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # ------------------------------------------------------------------ #
    # API
    # ------------------------------------------------------------------ #
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    # ------------------------------------------------------------------ #
    # LLM Provider
    # ------------------------------------------------------------------ #
    LLM_PROVIDER: str = "openrouter"      # "openrouter" | "openai" | "grok" | "anthropic" | "local"
    LLM_MODEL: str = "openai/gpt-4o-mini" # e.g. "openai/gpt-4o-mini", "anthropic/claude-3.5-sonnet", "google/gemini-2.0-flash-001"
    LLM_API_KEY: str = Field(default="", repr=False)
    OPENROUTER_API_KEY: str = Field(default="", repr=False)
    LLM_BASE_URL: str | None = None        # Custom endpoint override (defaults to https://openrouter.ai/api/v1 for openrouter)
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 4096
    LLM_TIMEOUT: int = 30                  # HTTP request timeout in seconds

    # ------------------------------------------------------------------ #
    # Query Understanding & Expansion (Phase 2.1)
    # ------------------------------------------------------------------ #
    QUERY_EXPANSION_MAX_QUERIES: int = 8   # Cap on generated expanded queries

    # ------------------------------------------------------------------ #
    # Embeddings
    # ------------------------------------------------------------------ #
    EMBEDDING_PROVIDER: str = "openai"    # e.g. "openai", "sentence-transformers"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_API_KEY: str = Field(default="", repr=False)
    EMBEDDING_BASE_URL: str | None = None  # OpenAI-compatible endpoint; defaults per provider (see embeddings factory)
    EMBEDDING_DIMENSION: int = 1536
    EMBEDDING_BATCH_SIZE: int = 32        # chunks per API request
    EMBEDDING_TIMEOUT: int = 60           # HTTP request timeout in seconds
    EMBEDDING_RETRY_COUNT: int = 5        # max retries on transient failures (429 / 5xx / network)
    EMBEDDING_RETRY_DELAY: float = 1.0    # base delay in seconds between retries
    EMBEDDING_MAX_BACKOFF: float = 60.0   # cap (seconds) on any single retry wait, incl. Retry-After

    # ------------------------------------------------------------------ #
    # BM25 (Phase 2.9)
    # ------------------------------------------------------------------ #
    BM25_INDEX_PATH: str = "data/artifacts/bm25"
    RETRIEVAL_MAX_PER_PAPER: int = 3      # cap hits per paper in a result list (0 = no cap)
    RETRIEVAL_CORPUS_CACHE_TTL: int = 900 # seconds a loaded run corpus is reused (0 = no cache)

    # ------------------------------------------------------------------ #
    # Vector Store
    # ------------------------------------------------------------------ #
    VECTOR_STORE: str = "faiss"           # "faiss" | "chroma" | "qdrant"
    VECTOR_STORE_PATH: str = "data/artifacts/vector_store"

    # ------------------------------------------------------------------ #
    # Relational / Metadata Store
    # ------------------------------------------------------------------ #
    DATABASE_URL: str = "sqlite:///./data/artifacts/ewrgv.db"
    SUPABASE_DATABASE_URL: str = Field(default="", repr=False)
    SUPABASE_URL: str = Field(default="", repr=False)
    SUPABASE_SERVICE_ROLE_KEY: str = Field(default="", repr=False)
    SUPABASE_STORAGE_BUCKET: str = "papers"

    # ------------------------------------------------------------------ #
    # Literature Collection (Phase 2.2)
    # ------------------------------------------------------------------ #

    # Comma-separated list of enabled providers: "semantic_scholar", "openalex"
    LITERATURE_PROVIDERS: str = "semantic_scholar,openalex"

    # Semantic Scholar
    SEMANTIC_SCHOLAR_API_KEY: str = Field(default="", repr=False)   # optional; anonymous OK
    SEMANTIC_SCHOLAR_BASE_URL: str = "https://api.semanticscholar.org/graph/v1"
    SEMANTIC_SCHOLAR_MAX_RESULTS: int = 20   # results per query

    # OpenAlex
    OPENALEX_BASE_URL: str = "https://api.openalex.org"
    OPENALEX_EMAIL: str = Field(default="", repr=False)   # optional polite-pool address
    OPENALEX_MAX_RESULTS: int = 20   # results per query

    # Unpaywall
    UNPAYWALL_BASE_URL: str = "https://api.unpaywall.org/v2"
    UNPAYWALL_EMAIL: str = Field(default="", repr=False)

    # Overall collection limits
    LITERATURE_REQUEST_TIMEOUT: int = 30    # HTTP timeout in seconds
    LITERATURE_MAX_TOTAL_RESULTS: int = 200  # hard cap across all queries + providers

    # ------------------------------------------------------------------ #
    # Chunking (Phase 2.7)
    # ------------------------------------------------------------------ #
    # Sizes are in approximate tokens (characters // 4).
    # These are initial reasonable defaults and should be tuned empirically.
    CHUNK_TARGET_SIZE: int = 500    # preferred tokens per chunk
    CHUNK_MAX_SIZE: int = 600       # hard upper bound tokens per chunk
    CHUNK_OVERLAP: int = 75         # overlap tokens between consecutive chunks
    CHUNK_MIN_SIZE: int = 20        # minimum tokens to emit a chunk
    CHUNK_MIN_WORDS: int = 60       # fold smaller sections / trailing chunks into neighbours (0 = off)

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #
    TOP_K: int = 20
    BM25_K: int = 20
    SEMANTIC_K: int = 20
    RERANK_TOP_N: int = 10

    # ------------------------------------------------------------------ #
    # Gap Detection
    # ------------------------------------------------------------------ #
    MAX_CANDIDATE_GAPS: int = 20
    TOP_K_GAPS: int = 5

    # ------------------------------------------------------------------ #
    # EWRGV Validation
    # ------------------------------------------------------------------ #
    MAX_VALIDATION_ITERATIONS: int = 3
    CONFIDENCE_THRESHOLD_VALID: float = 0.70
    CONFIDENCE_THRESHOLD_UNCERTAIN: float = 0.40


# Single shared instance – import this everywhere you need settings.
settings = Settings()
