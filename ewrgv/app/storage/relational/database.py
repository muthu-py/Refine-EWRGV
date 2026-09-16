"""
app/storage/relational/database.py
----------------------------------
Minimal asynchronous connection management for PostgreSQL via asyncpg.
"""

from typing import Optional
import asyncpg
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class DatabasePool:
    _pool: Optional[asyncpg.Pool] = None

    @classmethod
    async def get_pool(cls) -> asyncpg.Pool:
        """Get or initialize the asyncpg connection pool."""
        if cls._pool is None:
            if not settings.SUPABASE_DATABASE_URL:
                raise ValueError("SUPABASE_DATABASE_URL is not set in configuration.")
            
            logger.info("Initializing asyncpg connection pool to Supabase.")
            cls._pool = await asyncpg.create_pool(
                dsn=settings.SUPABASE_DATABASE_URL,
                min_size=1,
                max_size=10,
            )
        return cls._pool

    @classmethod
    async def close_pool(cls):
        """Close the asyncpg connection pool."""
        if cls._pool is not None:
            logger.info("Closing asyncpg connection pool.")
            await cls._pool.close()
            cls._pool = None


async def get_db_pool() -> asyncpg.Pool:
    """Dependency injector for getting the database pool."""
    return await DatabasePool.get_pool()
