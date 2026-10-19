"""
End-to-end retrieval verification over a real research run.

Loads the run's corpus (PaperChunks from chunks.json, stored pgvector
embeddings, persisted BM25 index) and prints sparse, dense and hybrid
results for a query:

    python scripts/verify_retrieval.py <run_id> ["query text"] [top_k]

Needs the database and Supabase Storage settings in .env.  Dense and hybrid
retrieval also need EMBEDDING_API_KEY; without it only BM25 is shown.
"""

import asyncio
import os
import sys

# Ensure the ewrgv package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.ingestion.corpus.run_corpus import RunCorpusLoader
from app.retrieval.service import RetrievalService
from app.storage.relational.database import DatabasePool, get_db_pool
from app.storage.repositories.embedding_repository import EmbeddingRepository
from app.storage.repositories.paper_repository import PaperRepository
from app.storage.supabase_storage import SupabaseStorageClient

DEFAULT_QUERY = "What are the limitations of current approaches?"


def _print(title, results):
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)
    for r in results:
        print(f"  Rank {r.rank}: [{r.chunk_id}] score={r.score:.4f}")
        print(f"    Paper: {r.paper_id} | Section: {r.section} [{r.section_type}] "
              f"| pages {r.page_start}-{r.page_end}")
        methods = [m["method"] for m in r.metadata.get("contributing_methods", [])]
        if methods:
            print(f"    Contributing methods: {methods}")
        print(f"    Text: {r.text[:80]}...")


async def main(run_id: str, query: str, top_k: int) -> None:
    pool = await get_db_pool()
    try:
        embedder = None
        if settings.EMBEDDING_API_KEY:
            from app.providers.embeddings.factory import create_embedding_provider

            embedder = create_embedding_provider()

        loader = RunCorpusLoader(
            paper_repo=PaperRepository(pool),
            storage_client=SupabaseStorageClient(),
            embedding_repo=EmbeddingRepository(pool) if embedder else None,
        )
        corpus = await loader.load(run_id)

        print("=" * 80)
        print("EWRGV — Retrieval verification")
        print("=" * 80)
        print(f"Run:    {run_id}")
        print(f"Query:  {query}")
        print(f"Top-K:  {top_k}")
        print(f"Corpus: {corpus.stats}")
        if not corpus.get_all_chunks():
            print("\nNo chunks for this run; run /chunks (and /represent) first.")
            return

        service = RetrievalService(corpus, embedding_provider=embedder)
        sparse = await service.aretrieve(query, top_k, method="sparse")
        _print("SPARSE RETRIEVAL (BM25)", sparse)

        if embedder is None:
            print("\nEMBEDDING_API_KEY not set — skipping dense and hybrid retrieval.")
            return

        dense = await service.aretrieve(query, top_k, method="dense")
        _print("DENSE RETRIEVAL (cosine similarity)", dense)
        hybrid = await service.aretrieve(query, top_k, method="hybrid")
        _print("HYBRID FUSION (Reciprocal Rank Fusion, k=60)", hybrid)

        ids = [r.chunk_id for r in hybrid]
        assert len(ids) == len(set(ids)), "FAIL: duplicates in hybrid results!"
        again = [r.chunk_id for r in await service.aretrieve(query, top_k, method="hybrid")]
        assert ids == again, "FAIL: non-deterministic results!"
        print("\n✓ No duplicates, deterministic ordering")
    finally:
        await DatabasePool.close_pool()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    asyncio.run(
        main(
            sys.argv[1],
            sys.argv[2] if len(sys.argv) > 2 else DEFAULT_QUERY,
            int(sys.argv[3]) if len(sys.argv) > 3 else 5,
        )
    )
