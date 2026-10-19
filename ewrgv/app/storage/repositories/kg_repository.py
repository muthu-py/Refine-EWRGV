"""
app/storage/repositories/kg_repository.py
-------------------------------------------
Persistence of run knowledge graphs (kg_graphs) and candidate-gap results
(gap_detection_results).  One row per run; a new build or detection run
replaces the previous one.
"""

from __future__ import annotations

import json
from typing import Optional

from asyncpg import Pool


def _load(value):
    return json.loads(value) if isinstance(value, str) else value


class KnowledgeGraphRepository:
    def __init__(self, db_pool: Pool) -> None:
        self._pool = db_pool

    async def save_graph(self, run_id: str, query_id: Optional[str], stats: dict, payload: dict) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO kg_graphs (run_id, query_id, built_at, stats, payload)
                VALUES ($1, $2, NOW(), $3::jsonb, $4::jsonb)
                ON CONFLICT (run_id) DO UPDATE SET
                    query_id = EXCLUDED.query_id, built_at = NOW(),
                    stats = EXCLUDED.stats, payload = EXCLUDED.payload
                """,
                run_id, query_id, json.dumps(stats), json.dumps(payload),
            )

    async def get_graph(self, run_id: str) -> Optional[dict]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT run_id, query_id, built_at, stats, payload FROM kg_graphs WHERE run_id = $1", run_id)
        if row is None:
            return None
        return {"run_id": row["run_id"], "query_id": row["query_id"], "built_at": row["built_at"].isoformat(),
                "stats": _load(row["stats"]), "payload": _load(row["payload"])}

    async def save_gaps(self, run_id: str, query_id: Optional[str], params: dict, stats: dict,
                        candidates: list, baseline: Optional[dict]) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO gap_detection_results (run_id, query_id, created_at, params, stats, candidates, baseline)
                VALUES ($1, $2, NOW(), $3::jsonb, $4::jsonb, $5::jsonb, $6::jsonb)
                ON CONFLICT (run_id) DO UPDATE SET
                    query_id = EXCLUDED.query_id, created_at = NOW(), params = EXCLUDED.params,
                    stats = EXCLUDED.stats, candidates = EXCLUDED.candidates, baseline = EXCLUDED.baseline,
                    validation_params = NULL, validation_stats = NULL, validation_results = NULL, validated_at = NULL
                """,
                run_id, query_id, json.dumps(params), json.dumps(stats), json.dumps(candidates),
                json.dumps(baseline) if baseline is not None else None,
            )

    async def get_gaps(self, run_id: str) -> Optional[dict]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT run_id, query_id, created_at, params, stats, candidates, baseline "
                "FROM gap_detection_results WHERE run_id = $1", run_id)
        if row is None:
            return None
        return {"run_id": row["run_id"], "query_id": row["query_id"], "created_at": row["created_at"].isoformat(),
                "params": _load(row["params"]), "stats": _load(row["stats"]),
                "candidates": _load(row["candidates"]), "baseline": _load(row["baseline"])}

    async def save_validations(self, run_id: str, params: dict, stats: dict, results: list[dict]) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute(
                "UPDATE gap_detection_results SET validation_params = $2::jsonb, "
                "validation_stats = $3::jsonb, validation_results = $4::jsonb, validated_at = NOW() "
                "WHERE run_id = $1",
                run_id, json.dumps(params), json.dumps(stats), json.dumps(results),
            )

    async def get_validations(self, run_id: str) -> Optional[dict]:
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT run_id, validated_at, validation_params, validation_stats, validation_results "
                "FROM gap_detection_results WHERE run_id = $1 AND validated_at IS NOT NULL", run_id)
        if row is None:
            return None
        return {"run_id": row["run_id"], "validated_at": row["validated_at"].isoformat(),
                "params": _load(row["validation_params"]), "stats": _load(row["validation_stats"]),
                "validation_results": _load(row["validation_results"])}
