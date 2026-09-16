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
        
    print(f"Connecting to Supabase at {settings.SUPABASE_DATABASE_URL.split('@')[1] if '@' in settings.SUPABASE_DATABASE_URL else 'localhost'}...")
    try:
        conn = await asyncpg.connect(settings.SUPABASE_DATABASE_URL)
        
        # Get total papers
        papers_count = await conn.fetchval("SELECT COUNT(*) FROM papers")
        
        # Get total sources
        sources_count = await conn.fetchval("SELECT COUNT(*) FROM paper_sources")
        
        print(f"\n==========================================")
        print(f"         DEDUPLICATION SUMMARY            ")
        print(f"==========================================")
        print(f"Total Unique Canonical Papers: {papers_count}")
        print(f"Total Discovery Sources (Provenance): {sources_count}")
        print(f"Total Duplicates Merged: {sources_count - papers_count}")
        print(f"==========================================")
        
        # Get papers with multiple sources
        duplicates_query = """
            SELECT p.title, COUNT(s.source_id) as source_count
            FROM papers p
            JOIN paper_sources s ON p.paper_id = s.paper_id
            GROUP BY p.paper_id, p.title
            HAVING COUNT(s.source_id) > 1
            ORDER BY source_count DESC
            LIMIT 10
        """
        multi_source_papers = await conn.fetch(duplicates_query)
        
        if multi_source_papers:
            print(f"\n--- TOP PAPERS MERGED FROM MULTIPLE SOURCES ---")
            for row in multi_source_papers:
                title = row['title']
                if len(title) > 80:
                    title = title[:77] + "..."
                print(f"- [Found {row['source_count']} times] {title}")
        else:
            print("\nNo merged papers found yet. (Every paper has exactly 1 source so far).")
            
        await conn.close()
    except Exception as e:
        print(f"Verification failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
