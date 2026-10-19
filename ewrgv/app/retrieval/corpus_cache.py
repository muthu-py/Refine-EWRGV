"""
app/retrieval/corpus_cache.py
-------------------------------
In-process cache of loaded RunCorpus objects for the retrieval endpoints.

Loading a run's corpus downloads every paper's chunks.json from storage and
every stored vector from the database; doing that on each /retrieve/* call
is slow and grows with the corpus.  Corpora are cached per
(run_id, embedding model) and expire after RETRIEVAL_CORPUS_CACHE_TTL seconds.

The /chunks and /represent workers call ``clear()`` when they finish, since
a paper can belong to several runs and re-chunking changes its chunk IDs.
Only the corpus is cached: a fresh RetrievalService is built per request,
so requests never share query-embedding state.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, Optional

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_CACHE: dict[tuple[str, Optional[str]], tuple[float, Any]] = {}
_LOCKS: dict[tuple[str, Optional[str]], asyncio.Lock] = {}


async def get_or_load(
    run_id: str,
    embedding_model: Optional[str],
    loader: Callable[[], Awaitable[Any]],
) -> Any:
    """Return the cached corpus for (run_id, model), loading it once if missing or expired."""
    key = (run_id, embedding_model)
    ttl = settings.RETRIEVAL_CORPUS_CACHE_TTL
    entry = _CACHE.get(key)
    if ttl > 0 and entry is not None and time.monotonic() - entry[0] < ttl:
        return entry[1]
    lock = _LOCKS.setdefault(key, asyncio.Lock())
    async with lock:
        entry = _CACHE.get(key)
        if ttl > 0 and entry is not None and time.monotonic() - entry[0] < ttl:
            return entry[1]
        corpus = await loader()
        if ttl > 0 and corpus.get_all_chunks():
            _CACHE[key] = (time.monotonic(), corpus)
            logger.info("Corpus cache: loaded run %s (model=%s)", run_id, embedding_model)
        return corpus


def invalidate(run_id: str) -> None:
    for key in [k for k in _CACHE if k[0] == run_id]:
        _CACHE.pop(key, None)


def clear() -> None:
    """Drop every cached corpus (called after /chunks and /represent finish)."""
    if _CACHE:
        logger.info("Corpus cache: cleared %d entries", len(_CACHE))
    _CACHE.clear()
