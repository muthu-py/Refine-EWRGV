"""
Run the LLM candidate-gap detector (app/gap_detection) on a processed run and
show its output.  This drives ResearchPipeline.run_on_corpus exactly as built:
Stage 4 retrieval -> Stage 6 LLM detection -> Stage 6b evidence scoring ->
Stage 7 ranking.  There is no API for this yet; this script is the way to see it.

    python scripts/run_llm_gaps.py <run_id> [options]

Options
    --query "..."      research question (default: the run's stored question)
    --query-id ID      use a stored /query result (question + expansions + understanding);
                       no extra LLM call for query understanding
    --no-expand        don't call query understanding; search the question only
    --hybrid           dense + BM25 retrieval (default: BM25 only, the pipeline's default)
    --chunks N         chunks sent to the LLM (pipeline retrieval_top_k, default 20)
    --top-k N          ranked gaps to show (default 5)
    --dry-run          stop before the LLM: print the chunks that would be sent
    --out FILE         JSON output path (default: data/artifacts/gaps/<run_id>_llm_gaps.json)

LLM calls: query understanding (unless --query-id / --no-expand) + one detection
call, through the LLM_PROVIDER / LLM_MODEL in .env.  Nothing is written to the
database.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.config import settings  # noqa: E402

for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def _wrap(text: str, indent: str = "     ") -> str:
    return textwrap.fill(" ".join(text.split()), width=100, initial_indent=indent, subsequent_indent=indent)


async def _load(run_id: str, hybrid: bool, query_id: str | None):
    from app.ingestion.corpus.run_corpus import RunCorpusLoader
    from app.storage.relational.database import get_db_pool
    from app.storage.repositories.embedding_repository import EmbeddingRepository
    from app.storage.repositories.paper_repository import PaperRepository
    from app.storage.supabase_storage import SupabaseStorageClient

    pool = await get_db_pool()
    question = await pool.fetchval("SELECT research_question FROM research_runs WHERE run_id::text = $1", run_id)
    if question is None:
        sys.exit(f"run_id {run_id} not found in research_runs")
    corpus = await RunCorpusLoader(
        paper_repo=PaperRepository(pool),
        storage_client=SupabaseStorageClient(),
        embedding_repo=EmbeddingRepository(pool) if hybrid else None,
    ).load(run_id)
    stored = None
    if query_id:
        from app.storage.repositories.query_repository import QueryRunRepository

        stored = await QueryRunRepository(pool).get(query_id)
        if stored is None:
            sys.exit(f"query_id {query_id} not found")
    return question, corpus, stored


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id")
    ap.add_argument("--query")
    ap.add_argument("--query-id")
    ap.add_argument("--no-expand", action="store_true")
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--chunks", type=int, default=20)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()

    from app.domain.models.query import ResearchQuery
    from app.domain.models.query_understanding import QueryUnderstanding
    from app.gap_detection.detector import GapDetector
    from app.gap_detection.evidence_scoring import CandidateEvidenceScorer
    from app.gap_detection.ranking import CandidateRanker
    from app.orchestration.research_pipeline import ResearchPipeline
    from app.retrieval.service import RetrievalService

    # A large prompt + up to MAX_CANDIDATE_GAPS structured answers can take > 30 s.
    settings.LLM_TIMEOUT = max(settings.LLM_TIMEOUT, 180)

    # One event loop for all database work (the asyncpg pool is bound to it).
    stored_question, corpus, stored = asyncio.run(_load(args.run_id, args.hybrid, args.query_id))
    stats = corpus.stats
    print(f"Run {args.run_id}: {stats['papers_with_chunks']} papers, {stats['chunks']} chunks, "
          f"{stats['embedded_chunks']} with vectors")

    # ---- the research query (question, expansions, understanding) ----
    if stored is not None:
        query = ResearchQuery(
            query=args.query or stored["research_question"],
            expanded_queries=list(stored["expanded_queries"]),
            understanding=QueryUnderstanding.model_validate(stored["understanding"]) if stored["understanding"] else None,
        )
    else:
        query = ResearchQuery(query=args.query or stored_question)
        if not args.no_expand and not args.dry_run:
            from app.api.routes.research import _get_qu_service

            qu = _get_qu_service()
            if qu is None:
                print("No LLM key configured: skipping query expansion.")
            else:
                print("Running query understanding (1 LLM call)...")
                query = qu.run(query)
    print(f"Question: {query.query}")
    print(f"Expanded queries ({len(query.expanded_queries)}): {query.expanded_queries}")

    embedder = None
    if args.hybrid:
        from app.providers.embeddings.factory import create_embedding_provider

        embedder = create_embedding_provider()

    def retriever_factory(c):
        return RetrievalService(c, embedding_provider=embedder)

    # ---- dry run: show Stage 4 output only ----
    if args.dry_run:
        pipeline = ResearchPipeline(None, None, None, None, None, None, None, None,
                                    retrieval_top_k=args.chunks, retriever_factory=retriever_factory)
        chunks = pipeline._retrieve_chunks(query, retriever_factory(corpus), corpus)
        print(f"\nDRY RUN: {len(chunks)} chunks would be sent to the LLM "
              f"({len({c.paper_id for c in chunks})} papers):")
        for i, c in enumerate(chunks, 1):
            paper = corpus.get_paper_by_id(c.paper_id)
            print(f"{i:2d}. [{c.section_type.value}] {c.section_title[:50]!r} | "
                  f"{(paper.title if paper else c.paper_id)[:60]}")
        return

    from app.providers.llm.factory import create_llm_provider

    if not (settings.LLM_API_KEY or settings.OPENROUTER_API_KEY):
        sys.exit("No LLM_API_KEY / OPENROUTER_API_KEY in .env; the LLM detector needs one.")
    print(f"LLM: {settings.LLM_PROVIDER} / {settings.LLM_MODEL}")

    pipeline = ResearchPipeline(
        query_understanding_service=None,
        literature_collector=None,
        document_parser=None,
        retriever=None,
        reranker=None,
        gap_detector=GapDetector(create_llm_provider(), max_candidates=settings.MAX_CANDIDATE_GAPS),
        candidate_ranker=CandidateRanker(),
        ewrgv_validator=None,
        top_k_gaps=args.top_k,
        retrieval_top_k=args.chunks,
        retriever_factory=retriever_factory,
        evidence_scorer=CandidateEvidenceScorer(),
    )
    print(f"Retrieving up to {args.chunks} chunks and running gap detection (1 LLM call)...")
    result = pipeline.run_on_corpus(query, corpus)
    if result.error:
        print(f"\nPipeline stopped at {result.stage_reached}: {result.error}")

    chunks_by_id = {c.chunk_id: c for c in result.retrieved_chunks}
    print(f"\nRetrieved {len(result.retrieved_chunks)} chunks "
          f"({len({c.paper_id for c in result.retrieved_chunks})} papers); "
          f"LLM proposed {len(result.candidate_gaps)} candidate gaps; top {len(result.top_k_gaps)}:\n")

    for gap in result.top_k_gaps:
        a = gap.evidence_assessment
        rk = gap.metadata.get("ranking", {})
        print(f"#{gap.rank}  [{gap.primary_gap_type.value}]  rank score {rk.get('adjusted_score')}  "
              f"(LLM confidence {gap.initial_confidence:.2f}, evidence {a.score if a else None} {a.status.value if a else ''})")
        print(_wrap(gap.gap_statement))
        if gap.metadata.get("rationale"):
            print(_wrap("why: " + gap.metadata["rationale"]))
        print(f"     papers: {len(gap.source_papers)}, supporting passages: {len(gap.relevant_passages)}, "
              f"contradicting: {len(gap.contradicting_passages)}")
        for cid in gap.relevant_passages[:3]:
            c = chunks_by_id.get(cid)
            if c:
                paper = corpus.get_paper_by_id(c.paper_id)
                print(f"       + [{c.section_type.value}] {(paper.title if paper else c.paper_id)[:70]}")
                print(_wrap(c.text[:220] + "...", indent="         "))
        print()

    others = [g for g in result.candidate_gaps if g.gap_id not in {t.gap_id for t in result.top_k_gaps}]
    if others:
        print("Other candidates (not in top-k):")
        for g in others:
            print(f"  - [{g.primary_gap_type.value}] conf {g.initial_confidence:.2f}: {g.gap_statement[:120]}")

    out = Path(args.out or Path(settings.ARTIFACTS_PATH if hasattr(settings, "ARTIFACTS_PATH") else "data/artifacts")
               / "gaps" / f"{args.run_id}_llm_gaps.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "run_id": args.run_id,
        "llm": f"{settings.LLM_PROVIDER}/{settings.LLM_MODEL}",
        "retrieval": "hybrid" if args.hybrid else "bm25",
        "query": query.model_dump(mode="json"),
        "retrieved_chunks": [
            {"chunk_id": c.chunk_id, "paper_id": c.paper_id, "section": c.section_title,
             "section_type": c.section_type.value, "text": c.text}
            for c in result.retrieved_chunks
        ],
        "candidate_gaps": [g.model_dump(mode="json") for g in result.candidate_gaps],
        "top_k_gaps": [g.model_dump(mode="json") for g in result.top_k_gaps],
        "error": result.error,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Full output (all candidates, evidence, scores): {out}")


if __name__ == "__main__":
    main()
