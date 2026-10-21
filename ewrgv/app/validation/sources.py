"""
app/validation/sources.py
---------------------------
Polite, cached access to the external literature for EWRGV validation and
temporal trends (free sources only):

    OpenAlex          works search, hit counts, counts per publication year
    Semantic Scholar  paper search (uses SEMANTIC_SCHOLAR_API_KEY when set)

Politeness
----------
* every response is cached: in process, and in Postgres
  (external_search_cache, migration 11) for EXTERNAL_CACHE_DAYS days, so
  re-running validation does not re-query the APIs;
* requests to one source are spaced (OpenAlex 0.15 s, S2 1.1 s) and
  serialised per source;
* 429 / 5xx / network errors back off exponentially (honouring Retry-After,
  capped at 30 s), up to 4 attempts; a source that keeps failing returns
  ``None`` and the caller records it as unavailable instead of failing.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import time
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

EXTERNAL_CACHE_DAYS = 7
_MIN_INTERVAL = {"openalex": 0.15, "semantic_scholar": 1.1}
_MAX_ATTEMPTS = 4


@dataclass
class ExternalItem:
    source: str
    item_id: str
    title: str
    abstract: str
    year: Optional[int]
    doi: Optional[str]
    url: Optional[str]
    citations: Optional[int]

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @property
    def text(self) -> str:
        return f"{self.title}. {self.abstract}".strip()


def _abstract_from_index(index: Optional[dict]) -> str:
    if not index:
        return ""
    positions = [(pos, word) for word, poss in index.items() for pos in poss]
    return " ".join(w for _, w in sorted(positions))


def phrase_query(terms: list[str]) -> str:
    """OpenAlex boolean query: every term as a quoted phrase, ANDed."""
    return " AND ".join(f'"{t.strip()}"' for t in terms if t and t.strip())


class ExternalLiterature:
    """
    Parameters
    ----------
    use_db_cache:
        Persist responses in external_search_cache (falls back to memory
        only if the database is unavailable).
    """

    def __init__(self, use_db_cache: bool = True, timeout: float = 30.0) -> None:
        self._use_db = use_db_cache
        self._timeout = timeout
        self._memory: dict[str, Any] = {}
        self._locks = {"openalex": asyncio.Lock(), "semantic_scholar": asyncio.Lock()}
        self._last_call = {"openalex": 0.0, "semantic_scholar": 0.0}
        self._client: Optional[httpx.AsyncClient] = None
        self.calls = {"openalex": 0, "semantic_scholar": 0, "cache_hits": 0, "failures": 0}

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def openalex_search(self, query: str, limit: int = 20) -> Optional[list[ExternalItem]]:
        data = await self._get("openalex", "/works", {
            "search": query, "per-page": min(limit, 50),
            "select": "id,doi,title,publication_year,abstract_inverted_index,cited_by_count",
        })
        if data is None:
            return None
        items = []
        for w in data.get("results") or []:
            items.append(ExternalItem(
                source="openalex", item_id=w.get("id") or "", title=w.get("title") or "",
                abstract=_abstract_from_index(w.get("abstract_inverted_index")),
                year=w.get("publication_year"), doi=(w.get("doi") or "").replace("https://doi.org/", "") or None,
                url=w.get("id"), citations=w.get("cited_by_count"),
            ))
        return items

    async def openalex_count(self, query: str) -> Optional[int]:
        data = await self._get("openalex", "/works", {"search": query, "per-page": 1, "select": "id"})
        if data is None:
            return None
        return int((data.get("meta") or {}).get("count") or 0)

    async def openalex_counts_by_year(self, query: str) -> Optional[dict[int, int]]:
        data = await self._get("openalex", "/works", {"search": query, "group_by": "publication_year"})
        if data is None:
            return None
        out: dict[int, int] = {}
        for g in data.get("group_by") or []:
            try:
                out[int(g["key"])] = int(g["count"])
            except (KeyError, TypeError, ValueError):
                continue
        return out

    async def s2_search(self, query: str, limit: int = 20) -> Optional[list[ExternalItem]]:
        data = await self._get("semantic_scholar", "/paper/search", {
            "query": query, "limit": min(limit, 100),
            "fields": "title,abstract,year,externalIds,citationCount,url",
        })
        if data is None:
            return None
        items = []
        for p in data.get("data") or []:
            ext = p.get("externalIds") or {}
            items.append(ExternalItem(
                source="semantic_scholar", item_id=p.get("paperId") or "", title=p.get("title") or "",
                abstract=p.get("abstract") or "", year=p.get("year"), doi=ext.get("DOI"),
                url=p.get("url"), citations=p.get("citationCount"),
            ))
        return items

    # ------------------------------------------------------------------ #
    # HTTP with cache, spacing and back-off
    # ------------------------------------------------------------------ #

    def _base(self, source: str) -> str:
        return settings.OPENALEX_BASE_URL.rstrip("/") if source == "openalex" else settings.SEMANTIC_SCHOLAR_BASE_URL.rstrip("/")

    async def _get(self, source: str, path: str, params: dict) -> Optional[dict]:
        key = hashlib.sha256(json.dumps([source, path, sorted(params.items())], default=str).encode()).hexdigest()
        if key in self._memory:
            self.calls["cache_hits"] += 1
            return self._memory[key]
        cached = await self._db_get(key)
        if cached is not None:
            self.calls["cache_hits"] += 1
            self._memory[key] = cached
            return cached

        headers = {}
        query = dict(params)
        if source == "openalex" and settings.OPENALEX_EMAIL:
            query["mailto"] = settings.OPENALEX_EMAIL
        if source == "semantic_scholar" and settings.SEMANTIC_SCHOLAR_API_KEY:
            headers["x-api-key"] = settings.SEMANTIC_SCHOLAR_API_KEY
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)

        async with self._locks[source]:
            for attempt in range(_MAX_ATTEMPTS):
                wait = _MIN_INTERVAL[source] - (time.monotonic() - self._last_call[source])
                if wait > 0:
                    await asyncio.sleep(wait)
                self._last_call[source] = time.monotonic()
                self.calls[source] += 1
                try:
                    resp = await self._client.get(self._base(source) + path, params=query, headers=headers)
                except httpx.HTTPError as exc:
                    logger.warning("%s request failed (attempt %d): %s", source, attempt + 1, type(exc).__name__)
                    await asyncio.sleep(min(30.0, 2 ** attempt + random.random()))
                    continue
                if resp.status_code == 200:
                    data = resp.json()
                    self._memory[key] = data
                    await self._db_put(key, source, data)
                    return data
                if resp.status_code in (429, 500, 502, 503, 504):
                    retry_after = resp.headers.get("Retry-After")
                    try:
                        delay = float(retry_after) if retry_after else 2 ** (attempt + 1)
                    except ValueError:
                        delay = 2 ** (attempt + 1)
                    logger.warning("%s returned %d (attempt %d); backing off %.1fs", source, resp.status_code, attempt + 1, min(delay, 30))
                    await asyncio.sleep(min(30.0, delay) + random.random() * 0.5)
                    continue
                logger.warning("%s returned %d for %s", source, resp.status_code, path)
                break
        self.calls["failures"] += 1
        return None

    async def _db_get(self, key: str) -> Optional[dict]:
        if not self._use_db:
            return None
        try:
            from app.storage.relational.database import get_db_pool

            pool = await get_db_pool()
            row = await pool.fetchrow(
                "SELECT response FROM external_search_cache WHERE key = $1 "
                "AND fetched_at > NOW() - make_interval(days => $2)", key, EXTERNAL_CACHE_DAYS)
            if row is None:
                return None
            value = row["response"]
            return json.loads(value) if isinstance(value, str) else value
        except Exception as exc:  # noqa: BLE001 - cache is best effort
            logger.debug("external cache read failed: %s", exc)
            return None

    async def _db_put(self, key: str, source: str, data: dict) -> None:
        if not self._use_db:
            return
        try:
            from app.storage.relational.database import get_db_pool

            pool = await get_db_pool()
            await pool.execute(
                "INSERT INTO external_search_cache (key, source, response, fetched_at) VALUES ($1, $2, $3::jsonb, NOW()) "
                "ON CONFLICT (key) DO UPDATE SET response = EXCLUDED.response, fetched_at = NOW()",
                key, source, json.dumps(data))
        except Exception as exc:  # noqa: BLE001
            logger.debug("external cache write failed: %s", exc)
