import asyncio
import sys
from pathlib import Path

# Add project root to path so we can import app modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings
import asyncpg

async def main():
    if not settings.SUPABASE_DATABASE_URL:
        print("Error: SUPABASE_DATABASE_URL is not set in your .env file.")
        sys.exit(1)
        
    migration_file = Path(__file__).resolve().parent / "migrations" / "01_create_papers_table.sql"
    
    if not migration_file.exists():
        print(f"Error: Migration file not found at {migration_file}")
        sys.exit(1)
        
    sql = migration_file.read_text(encoding="utf-8")
    
    print(f"Connecting to database at {settings.SUPABASE_DATABASE_URL.split('@')[1] if '@' in settings.SUPABASE_DATABASE_URL else 'localhost'}...")
    try:
        conn = await asyncpg.connect(settings.SUPABASE_DATABASE_URL)
        print("Connected successfully. Running migration...")
        await conn.execute(sql)
        print("Migration executed successfully! Tables 'papers' and 'paper_sources' have been created.")
        await conn.close()
    except Exception as e:
        print(f"Migration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
