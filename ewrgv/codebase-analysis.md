# EWRGV codebase analysis

Analysed: `Refine-ResearchGap/ewrgv` (about 12.6k lines of Python under `app/`, 424 tests). Date: 2026-10-17.

## Summary

The pipeline is built and tested from the research question up to embeddings and a BM25 index (phases 2.1 to 2.9). Everything after that is a skeleton: retrieval, gap detection, and the whole EWRGV validation stage, including the confidence formula. The validation stage, which is the research contribution, has no logic yet. Every method in `app/validation/`, `app/evidence/`, `app/coverage/`, `app/scoring/confidence.py` and `app/explanation/` logs a warning and returns an empty or default value.

All 424 tests pass (`pytest tests -q`, Python 3.13, in about 3 seconds). The tests mock the database and HTTP calls, so none of them touch Supabase, OpenRouter, Semantic Scholar or OpenAlex.

## Architecture

The code is a modular monolith built on FastAPI. Dependencies point one way: API, then orchestration, then the application modules, then the interfaces in `domain/interfaces.py`, then providers and storage. Data is stored in Supabase Postgres through `asyncpg`, with `pgvector` for embeddings. PDFs go to Supabase Storage. Job progress is held in memory in `core/job_store.py`.

| Stage | Module | State |
|---|---|---|
| 2.1 Query understanding and expansion (LLM) | `query_understanding/`, `providers/llm/` | Implemented: `POST /research/query` |
| 2.2 Literature collection (Semantic Scholar and OpenAlex) | `ingestion/collectors/`, `providers/search/` | Implemented: `POST /research/collect` |
| 2.3 Deduplication | `ingestion/deduplication/` | Implemented, runs after collect |
| 2.4 Open-access discovery and PDF download (Unpaywall) | `ingestion/acquisition/` | Implemented: `POST /research/acquire` |
| 2.5 PDF parsing (PyMuPDF) | `ingestion/parsers/` | Implemented: `POST /research/parse` |
| 2.6 Sectioning (heading detection and classification) | `ingestion/sectioning/` | Implemented: `POST /research/sections` |
| 2.7 Chunking | `ingestion/chunking/` | Implemented: `POST /research/chunks` |
| 2.8 Corpus health | `storage/corpus_health.py` | Implemented: `GET /research/corpus/health` |
| 2.9 Dense and sparse representation | `ingestion/representation/`, `retrieval/bm25_index.py` | Implemented: `POST /research/represent` |
| Retrieval (semantic search, RRF fusion, reranking) | `retrieval/semantic.py`, `fusion.py`, `reranking.py`, `bm25.py` | Skeleton, returns `[]` |
| Gap detection, representation and ranking | `gap_detection/` | Skeleton |
| EWRGV validation | `validation/`, `evidence/`, `coverage/`, `scoring/`, `explanation/` | Skeleton |
| End-to-end orchestration | `orchestration/research_pipeline.py` | Runs stages 1 and 2 only |

`/research/gaps` and `/research/validate` return `status: "not_implemented"`.

## How to run

```bash
cd ewrgv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# create .env (there is no .env.example, despite the README) with at least:
#   OPENROUTER_API_KEY=...      (query understanding)
#   SUPABASE_DATABASE_URL=...   (every endpoint after /collect)
#   SUPABASE_URL=... SUPABASE_SERVICE_ROLE_KEY=...  (PDF storage)
#   EMBEDDING_API_KEY=...       (OpenAI embeddings)
#   UNPAYWALL_EMAIL=...
# apply scripts/migrations/01..07 *manually* (see bug 3)
uvicorn app.main:app --reload       # docs at http://localhost:8000/docs
pytest tests -q
```

The intended flow is `query`, `collect` (which returns a `run_id`), `acquire`, `parse`, `sections`, `chunks`, `represent`, polling each job's status endpoint along the way.

## Bugs found

1. **The representation error path crashes.** In `api/routes/research.py:1074-1077`, the `except` block of `start_representation` uses `JobStatus` and `HTTPException`, but neither is imported in that function or at the top of the module. Any setup failure (for example, a missing DB URL or embedding key) raises a `NameError` instead of returning the intended 500 with a useful message.
2. **Background jobs can be silently garbage-collected.** All five `asyncio.create_task(...)` calls (`research.py:484, 604, 714, 812, 1065`) drop the returned task. asyncio keeps only a weak reference, so a long acquisition or embedding job can disappear mid-run, and its job would stay `RUNNING` forever. The fix is to keep the tasks in a set on `app.state` and discard each one when it finishes.
3. **`scripts/run_migrations.py` only applies migration 01.** The file path is hard-coded to `01_create_papers_table.sql`. A fresh database following the README would be missing the fulltext, parsing, section, chunk, constraint and embedding columns and tables from migrations 02 to 07.
4. **Blocking I/O runs inside async handlers.** `/query` and `/collect` call the synchronous LLM client and `httpx.Client`, which use `time.sleep` for retries (`collectors/http_client.py:104,121`), directly in `async def` routes. While one collect runs, the event loop is blocked, so every running background job and every status poll stalls with it. These calls should be wrapped in `run_in_threadpool`, or the clients made async.
5. **A failed persist in `/collect` still looks like success.** If the Supabase write fails, the error is only logged and the response still returns a `run_id`. The next call, `/acquire`, then returns 404 for that `run_id`, and nothing points the caller to the real cause.
6. **Deduplication runs over the whole database on every collect** (`Deduplicator.run()` with no run scope). This gets slower as the corpus grows, and it can merge papers across unrelated research runs.
7. **The job store is in memory only.** A server restart loses every job ID, and the asyncpg pool is never closed on shutdown (`close_pool` exists but `main.py`'s lifespan doesn't call it).
8. **The config and the code disagree.** `VECTOR_STORE="faiss"` and `DATABASE_URL=sqlite://...` are never used, because the code uses pgvector and Supabase. `EMBEDDING_DIMENSION` is configurable, but the migration hard-codes `vector(1536)`, so switching the embedding model breaks inserts.
9. **There are two chunk models.** `DocumentChunk` (in `domain/models/paper.py`) is what retrieval, gap detection and validation expect. `PaperChunk` (in `domain/models/paper_chunk.py`) is what chunking actually produces. Nothing converts one into the other.
10. **The README is out of date.** It says retrieval, collectors and providers are "not started" and refers to a `.env.example` that doesn't exist.

## Validation stage: design weak spots

These are the issues to settle before writing the code, because they decide whether EWRGV can produce meaningful results.

1. **Validation searches the same corpus that detection used.** `EWRGVValidator` gets the same `retriever` as the rest of the pipeline. Running direct, terminology, method and counter searches over the same chunks that produced the gap mostly re-finds the evidence the detector already saw. Real counter-evidence usually sits in papers that were never collected. The counter and terminology searches should go back out to Semantic Scholar and OpenAlex (and run them through acquisition, parsing and chunking, or at least use abstracts) before searching locally. This is the most important design decision for the contribution.
2. **The default confidence makes every gap UNSUPPORTED.** `EWRGVConfidenceScorer.score` returns `0.0`, and `classify_by_confidence(0.0)` gives `UNSUPPORTED_GAP`. If the validator is wired up before the formula exists, every gap will be rejected without any error.
3. **One score can't tell "no evidence found" apart from "evidence says it's open".** A single score with two thresholds maps both "nothing found because the corpus is thin" and "strong evidence the gap is open" onto the same scale. `CoverageAssessment` already has `corpus_adequacy` and `corpus_uncertainty` fields. Use them as a gate: low adequacy should force `UNCERTAIN_GAP` whatever the evidence score is, rather than being one weighted term among others.
4. **The evidence labels overlap.** SUPPORTING is defined as "confirms the gap exists". In practice, a passage can only support a gap when it is a stated limitation or future-work note (the `LIMITATIONS` and `FUTURE_WORK` sections are already classified). COUNTER ("already addressed") and CONTRADICTORY ("contradicts the gap claim") are hard for an LLM to separate. You could merge them, or redefine CONTRADICTORY to mean conflicting findings, which matches the `CONTRADICTORY_EVIDENCE` gap type.
5. **There is no stopping rule for the iteration loop.** `should_iterate` always returns `False`, and nothing defines when coverage counts as sufficient. A concrete rule would be to iterate while `retrieval_coverage` is below a threshold, or while the last round found new COUNTER evidence, up to `MAX_VALIDATION_ITERATIONS`.
6. **Evidence weighting needs inputs the code doesn't keep.** The aggregator is meant to weight evidence by search type and relevance, but `Evidence` doesn't carry the paper's year, venue or citation count. That data exists on `Paper` and `paper_sources` and could weight recency and credibility. A recent COUNTER paper should count for more than an old one.
7. **Gap detection doesn't feed the validator what it needs.** The query generator depends on `StructuredGapRepresentation` (`key_concepts`, `synonyms`, `methodology`), but nothing fills it in yet. Query understanding already produces `terminology_variants` and `method_variants`, which could seed it.
8. **There are no evaluation hooks.** There is no labelled set of known-open and known-closed gaps, and no baseline (detection without validation) to compare against. The roadmap lists both under phase 5, but the confidence formula can't be tuned without them, so they should come earlier.

## Suggested next steps, in order

1. Fix bugs 1 to 3, which are small and easy to isolate.
2. Implement `reciprocal_rank_fusion` (about 10 lines) and `SemanticRetriever` against pgvector, and add a `PaperChunk` to `DocumentChunk` adapter.
3. Implement gap detection over chunks from the `LIMITATIONS`, `FUTURE_WORK` and `DISCUSSION` sections first, filling in `StructuredGapRepresentation`.
4. Design the validator around external counter-search plus a corpus-adequacy gate, and build a small labelled evaluation set alongside it.
