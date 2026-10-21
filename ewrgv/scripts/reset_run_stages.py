"""
Reset a run's papers so /sections, /chunks and /represent process them again
(after changing heading detection or chunking).  Each stage only picks up
papers it has not processed yet, so without this a re-run does nothing.

    python scripts/reset_run_stages.py <run_id> [--from sections|chunks|represent]

Only status columns are reset; no data is deleted.  Stale vectors of old
chunks are pruned by /represent when it re-embeds the new chunks.
Note: papers shared with other runs are reset for those runs too.
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncpg

from app.core.config import settings

RESETS = {
    "sections": "section_status = 'NOT_EXTRACTED', chunk_status = 'NOT_CHUNKED', embedding_status = 'NOT_EMBEDDED'",
    "chunks": "chunk_status = 'NOT_CHUNKED', embedding_status = 'NOT_EMBEDDED'",
    "represent": "embedding_status = 'NOT_EMBEDDED'",
}
GUARD = {
    "sections": "COALESCE(parsing_status, '') = 'PARSED'",
    "chunks": "COALESCE(section_status, '') = 'EXTRACTED'",
    "represent": "COALESCE(chunk_status, '') = 'CHUNKED'",
}


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--from", dest="stage", choices=list(RESETS), default="sections")
    args = parser.parse_args()
    if not settings.SUPABASE_DATABASE_URL:
        sys.exit("SUPABASE_DATABASE_URL is not set in .env")
    conn = await asyncpg.connect(settings.SUPABASE_DATABASE_URL)
    try:
        status = await conn.execute(
            f"UPDATE papers SET {RESETS[args.stage]} "
            f"WHERE {GUARD[args.stage]} AND paper_id IN "
            "(SELECT paper_id FROM paper_sources WHERE run_id = $1)",
            args.run_id,
        )
        print(f"reset from {args.stage}: {status.split()[-1]} papers")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
