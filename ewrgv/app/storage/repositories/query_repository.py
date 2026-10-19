"""
app/storage/repositories/query_repository.py
-----------------------------------------------
Persistence for POST /research/query results (table ``query_runs``).
"""

from __future__ import annotations

import json
from typing import Optional

from asyncpg import Pool

from app.domain.models.query import ResearchQuery


class QueryRunRepository:
    """Stores and loads query understanding + expanded queries by query_id."""

    def __init__(self, db_pool: Pool) -> None:
        self._pool = db_pool

    async def save(self, query: ResearchQuery) -> None:
        understanding = (
            query.understanding.model_dump(mode="json") if query.understanding is not None else None
        )
        query_type = getattr(query.query_type, "value", str(query.query_type))
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO query_runs (query_id, research_question, query_type, understanding, expanded_queries)
                VALUES ($1, $2, $3, $4::jsonb, $5::jsonb)
                ON CONFLICT (query_id) DO UPDATE SET
                    research_question = EXCLUDED.research_question,
                    query_type        = EXCLUDED.query_type,
                    understanding     = EXCLUDED.understanding,
                    expanded_queries  = EXCLUDED.expanded_queries
                """,
                query.query_id,
                query.query,
                query_type,
                json.dumps(understanding) if understanding is not None else None,
                json.dumps(list(query.expanded_queries)),
            )

    async def get(self, query_id: str) -> Optional[dict]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT query_id::text AS query_id, research_question, query_type,
                       understanding, expanded_queries
                FROM query_runs WHERE query_id::text = $1
                """,
                query_id,
            )
        if row is None:
            return None
        out = dict(row)
        for key in ("understanding", "expanded_queries"):
            if isinstance(out.get(key), str):
                out[key] = json.loads(out[key])
        out["expanded_queries"] = out.get("expanded_queries") or []
        return out
