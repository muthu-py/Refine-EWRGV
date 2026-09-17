"""
app/storage/repositories/embedding_repository.py
------------------------------------------------
Repository for managing dense embeddings in PostgreSQL via pgvector.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

from asyncpg import Pool

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class EmbeddingRecord:
    """Domain model for a single chunk embedding record."""
    chunk_id: str
    paper_id: str
    embedding: list[float]
    embedding_model: str
    embedding_dimension: int


class EmbeddingRepository:
    """
    Handles database operations for the chunk_embeddings table.

    Requires pgvector extension to be enabled in PostgreSQL.
    """

    def __init__(self, db_pool: Pool):
        self._pool = db_pool

    async def save_embeddings(self, records: list[EmbeddingRecord]) -> None:
        """
        Batch upsert embeddings into the database.

        Idempotent: if (chunk_id, embedding_model) already exists, it is updated.
        """
        if not records:
            return

        query = """
            INSERT INTO chunk_embeddings (
                chunk_id,
                paper_id,
                embedding,
                embedding_model,
                embedding_dimension,
                updated_at
            )
            VALUES ($1, $2, $3::vector, $4, $5, CURRENT_TIMESTAMP)
            ON CONFLICT (chunk_id, embedding_model) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                embedding_dimension = EXCLUDED.embedding_dimension,
                updated_at = CURRENT_TIMESTAMP
        """

        # asyncpg executemany expects a list of tuples
        values = [
            (
                r.chunk_id,
                r.paper_id,
                json.dumps(r.embedding),  # passing as JSON string which pgvector can cast
                r.embedding_model,
                r.embedding_dimension,
            )
            for r in records
        ]

        async with self._pool.acquire() as conn:
            await conn.executemany(query, values)
            logger.debug(
                "Saved %d embeddings for model %s",
                len(records),
                records[0].embedding_model if records else "unknown",
            )

    async def get_embedded_chunk_ids(self, paper_id: str, embedding_model: str) -> set[str]:
        """
        Retrieve all chunk IDs for a paper that already have embeddings for the given model.

        Used for incremental embedding (skipping already-processed chunks).
        """
        query = """
            SELECT chunk_id
            FROM chunk_embeddings
            WHERE paper_id = $1 AND embedding_model = $2
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, paper_id, embedding_model)
            return {row["chunk_id"] for row in rows}

    async def get_embedding_counts(self, paper_id: str, embedding_model: str) -> int:
        """
        Get the total number of embeddings for a specific paper and model.
        """
        query = """
            SELECT COUNT(*)
            FROM chunk_embeddings
            WHERE paper_id = $1 AND embedding_model = $2
        """
        async with self._pool.acquire() as conn:
            count = await conn.fetchval(query, paper_id, embedding_model)
            return count or 0

    async def delete_embeddings(self, chunk_ids: list[str], embedding_model: str) -> None:
        """
        Delete specific embeddings. Useful for rebuilds or cleanup.
        """
        if not chunk_ids:
            return

        query = """
            DELETE FROM chunk_embeddings
            WHERE chunk_id = ANY($1) AND embedding_model = $2
        """
        async with self._pool.acquire() as conn:
            await conn.execute(query, chunk_ids, embedding_model)
