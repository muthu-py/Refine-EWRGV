"""
Run your friend's LLM gap detector once on a processed run and print the output.

    python scripts/run_llm_gap_baseline.py <run_id> "<research question>" [--chunks 30] [--all-sections]

Steps (no database writes, no code changes):
  1. Query understanding on the question (1 LLM call) -> expanded queries
  2. Load the run corpus (chunks.json + stored embeddings)
  3. Hybrid retrieval for the question + every expanded query, keep only
     gap-evidence sections (unless --all-sections), fuse with RRF, max 3 per paper
  4. GapDetector.detect (1 LLM call)
  5. CandidateEvidenceScorer + CandidateRanker -> Top-K
Prints the candidates and writes everything to gap_baseline_<run_id>.json.
"""

import argparse
import asyncio
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.domain.enums import SectionType
from app.domain.models.query import ResearchQuery
from app.gap_detection.detector import GapDetector
from app.gap_detection.evidence_scoring import CandidateEvidenceScorer
from app.gap_detection.ranking import CandidateRanker
from app.ingestion.corpus.run_corpus import RunCorpusLoader
from app.providers.embeddings.factory import create_embedding_provider
from app.providers.llm.factory import create_llm_provider
from app.query_understanding.service import QueryUnderstandingService
from app.retrieval.service import RetrievalService
from app.storage.relational.database import DatabasePool, get_db_pool
from app.storage.repositories.embedding_repository import EmbeddingRepository
from app.storage.repositories.paper_repository import PaperRepository
from app.storage.supabase_storage import SupabaseStorageClient

GAP_EVIDENCE = {
    SectionType.LIMITATIONS, SectionType.FUTURE_WORK, SectionType.DISCUSSION,
    SectionType.CONCLUSION, SectionType.RESULTS,
}
NOISE = {SectionType.REFERENCES}
RRF_K = 60
PER_QUERY = 30
MAX_PER_PAPER = 3


def _keep(chunk, all_sections: bool) -> bool:
    if chunk.section_type in NOISE:
        return False
    return all_sections or chunk.section_type in GAP_EVIDENCE


async def main(run_id: str, question: str, n_chunks: int, all_sections: bool) -> None:
    llm = create_llm_provider()

    print("1/5 Query understanding ...")
    query = await asyncio.to_thread(
        QueryUnderstandingService(llm=llm, max_expanded_queries=settings.QUERY_EXPANSION_MAX_QUERIES).run,
        ResearchQuery(query=question),
    )
    queries = [q for q in dict.fromkeys([question, *query.expanded_queries]) if q.strip()]
    print(f"    {len(queries)} queries")

    pool = await get_db_pool()
    try:
        print("2/5 Loading run corpus ...")
        embedder = create_embedding_provider()
        corpus = await RunCorpusLoader(
            paper_repo=PaperRepository(pool),
            storage_client=SupabaseStorageClient(),
            embedding_repo=EmbeddingRepository(pool),
        ).load(run_id)
        print(f"    {corpus.stats['papers_with_chunks']} papers, {corpus.stats['chunks']} chunks")
        if not corpus.get_all_chunks():
            print("No chunks for this run; run /chunks and /represent first.")
            return

        print("3/5 Hybrid retrieval over all queries ...")
        service = RetrievalService(corpus, embedding_provider=embedder)
        fused: dict[str, float] = defaultdict(float)
        matched: dict[str, list[str]] = defaultdict(list)
        for q in queries:
            results = await service.aretrieve(q, PER_QUERY * 3, method="hybrid")
            kept = [r for r in results if _keep(corpus.get_chunk_by_id(r.chunk_id), all_sections)]
            for rank, r in enumerate(kept[:PER_QUERY], start=1):
                fused[r.chunk_id] += 1.0 / (RRF_K + rank)
                matched[r.chunk_id].append(q)
        per_paper: dict[str, int] = defaultdict(int)
        chunks = []
        for chunk_id in sorted(fused, key=lambda c: (-fused[c], c)):
            chunk = corpus.get_chunk_by_id(chunk_id)
            if per_paper[chunk.paper_id] >= MAX_PER_PAPER:
                continue
            per_paper[chunk.paper_id] += 1
            chunks.append(chunk)
            if len(chunks) >= n_chunks:
                break
        print(f"    {len(chunks)} chunks from {len(per_paper)} papers sent to the detector")

        print("4/5 LLM gap detection ...")
        detector = GapDetector(llm, max_candidates=settings.MAX_CANDIDATE_GAPS, max_chunks=n_chunks)
        candidates = await asyncio.to_thread(detector.detect, query, chunks)
        print(f"    {len(candidates)} grounded candidates")

        print("5/5 Evidence scoring + ranking ...")
        scored = CandidateEvidenceScorer().score_all(candidates, chunks, paper_lookup=corpus.get_paper_by_id)
        top = CandidateRanker().rank(scored, top_k=settings.TOP_K_GAPS)
    finally:
        await DatabasePool.close_pool()

    chunk_by_id = {c.chunk_id: c for c in chunks}

    def title(paper_id):
        paper = corpus.get_paper_by_id(paper_id)
        return paper.title if paper else paper_id

    print("\n" + "=" * 80)
    print(f"TOP {len(top)} OF {len(candidates)} CANDIDATE GAPS")
    print("=" * 80)
    for gap in top:
        ea = gap.evidence_assessment
        print(f"\n#{gap.rank} [{gap.primary_gap_type.value}] {gap.gap_statement}")
        print(f"   LLM confidence {gap.initial_confidence:.2f} | evidence score "
              f"{ea.score:.2f} ({ea.status.value}) | ranking score "
              f"{gap.metadata['ranking']['adjusted_score']:.3f}")
        if gap.metadata.get("rationale"):
            print(f"   Why: {gap.metadata['rationale']}")
        for cid in gap.relevant_passages:
            c = chunk_by_id.get(cid)
            if c:
                print(f"   + {c.section_title} [{c.section_type.value}] p{c.page_start}: {title(c.paper_id)[:70]}")
        for cid in gap.contradicting_passages:
            c = chunk_by_id.get(cid)
            if c:
                print(f"   - {c.section_title} [{c.section_type.value}] p{c.page_start}: {title(c.paper_id)[:70]}")

    out = f"gap_baseline_{run_id}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id,
            "question": question,
            "queries": queries,
            "chunks_sent": [
                {"chunk_id": c.chunk_id, "paper": title(c.paper_id), "section": c.section_title,
                 "section_type": c.section_type.value, "matched_queries": matched[c.chunk_id]}
                for c in chunks
            ],
            "all_candidates": [g.model_dump(mode="json") for g in scored],
            "top_k": [g.model_dump(mode="json") for g in top],
        }, f, indent=2, ensure_ascii=False)
    print(f"\nFull output written to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_id")
    p.add_argument("question")
    p.add_argument("--chunks", type=int, default=30, help="chunks sent to the LLM (default 30)")
    p.add_argument("--all-sections", action="store_true", help="don't restrict to gap-evidence sections")
    a = p.parse_args()
    asyncio.run(main(a.run_id, a.question, a.chunks, a.all_sections))
