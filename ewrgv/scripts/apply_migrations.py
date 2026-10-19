"""
Apply the pending schema migrations (04, 05, 07) to the Supabase database
in SUPABASE_DATABASE_URL.  Every statement uses IF NOT EXISTS, so re-running
is safe.  06 is skipped: it uses ADD CONSTRAINT IF NOT EXISTS, which
PostgreSQL does not support.

    python scripts/apply_migrations.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncpg

from app.core.config import settings

MIGRATIONS = [
    "04_add_section_columns.sql",
    "05_add_chunk_columns.sql",
    "07_add_embedding_table.sql",
]


async def main() -> None:
    if not settings.SUPABASE_DATABASE_URL:
        sys.exit("SUPABASE_DATABASE_URL is not set in .env")
    conn = await asyncpg.connect(settings.SUPABASE_DATABASE_URL)
    try:
        for name in MIGRATIONS:
            sql = (Path(__file__).resolve().parent / "migrations" / name).read_text(
                encoding="utf-8-sig"  # strips the BOM some files start with
            )
            await conn.execute(sql)
            print(f"applied {name}")
        cols = await conn.fetch(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'papers' AND column_name = ANY($1)",
            ["section_status", "chunk_status", "chunks_storage_path", "embedding_status"],
        )
        table = await conn.fetchval("SELECT to_regclass('public.chunk_embeddings')")
        print("papers columns present:", sorted(r["column_name"] for r in cols))
        print("chunk_embeddings table:", table)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
