# Plan: one-call research pipeline up to retrieval

Status: **proposal, nothing implemented.** Based on branch `integrate/gap-detection`
plus the uncommitted fixes on your laptop (local embeddings, chunk merging,
per-paper cap).

## Goal

Before: you call 8 endpoints by hand, copy `expanded_queries` from `/query`
into `/collect`, copy the `run_id` into every later call, and poll each job
until it finishes.

After: one call, `POST /research/run {"query": "..."}`, returns a `run_id`
straight away and drives query understanding → collect → acquire → parse →
sections → chunks → represent in the background.  `GET /research/run/{run_id}`
shows which stage it is on with counts.  Retrieval then runs over the original
question *and* all expanded queries.  The existing endpoints keep working for
manual testing.

---

## 1. Manual handoffs and logic gaps found

Checked in the code (file references are on the branch):

| # | Problem | Where | Effect |
|---|---------|-------|--------|
| 1 | `/query` and `/collect` aren't connected: you paste `expanded_queries` yourself | `research.py` `/query`, `/collect` | the avalanche you described doesn't exist |
| 2 | `/query`'s `query_id` is thrown away; `/collect` makes a new id that becomes the `run_id` | `/collect` builds a fresh `ResearchQuery` | the run can't be traced back to its query understanding |
| 3 | Query understanding (domain, methods, datasets…) is never saved | `research_runs` has only `run_id, research_question, created_at` | later stages (gap detection) can't use it |
| 4 | `/collect` searches only the expanded queries; the original question is dropped when expansions exist | `collectors/service.py:105` `queries = expanded or [query]` | misses papers that match the question as written |
| 5 | `/collect` and `/query` run blocking HTTP/LLM calls inside async routes | `service.collect()` uses sync `httpx.Client`; `qu_service.run()` is sync | the whole server freezes during collection (other requests and job polling stall) |
| 6 | `/collect` returns 200 and a `run_id` even when saving fails, and when 0 papers are found no `research_runs` row is written | `/collect` `except Exception: logger.error` | `/acquire` then 404s on a run_id you were given (what happened earlier) |
| 7 | Deduplication scans the entire `papers` table after every collect | `Deduplicator.run()` | gets slower with every run |
| 8 | acquire → parse → sections → chunks → represent each need a manual start and polling | five `asyncio.create_task` routes | nothing chains |
| 9 | Each stage only picks papers it hasn't processed; re-running needs `reset_run_stages.py` | `get_papers_for_*` filters | fine for a pipeline, but failed downloads (`DOWNLOAD_FAILED`) are never retried |
| 10 | Job status lives in memory only, and the background tasks aren't kept referenced | `job_store.py`, `create_task(...)` with no reference | a restart loses all progress; Python may garbage-collect a running task |
| 11 | Nothing stops two jobs of the same stage running on one run at once | routes | double downloads / duplicate chunks |
| 12 | `/retrieve/*` takes one query; the expanded queries are not used at retrieval time | `RetrieveRequest` | ResearchPipeline Stage 4 does multi-query retrieval but isn't exposed |
| 13 | Reference-list chunks (74 in your run) are searchable | `RetrievalService` | noise in results and in gap detection |
| 14 | `ResearchPipeline` is sync, not wired to any route, and would re-run collection | `orchestration/research_pipeline.py` | `/gaps` and `/validate` are still placeholders |
| 15 | Acquisition yield is low: 24 PDFs from 141 papers (64 failed, 53 not open access), with no failure reason per paper | acquisition discovery | small corpus |
| 16 | `/collect` ignores `max_results_per_query` from the request and uses `SEMANTIC_SCHOLAR_MAX_RESULTS` from `.env`; a silent total cap (`LITERATURE_MAX_TOTAL_RESULTS`) applies on top | `research.py:339` | why you got exactly 200 papers |
| 17 | Stage statuses are stored per paper, not per run | `papers.*_status` columns | a paper shared with an older run is skipped or reprocessed unpredictably (why this run processed 32 papers, not 24) |
| 18 | Every retrieval request re-downloads all `chunks.json` files and vectors for the run | `_build_retrieval_service` → `RunCorpusLoader.load` | slow, worse as the corpus grows |
| 19 | No filter by section type | `RetrieveRequest` | can't ask for only Limitations / Discussion / Conclusion / Future Work |

---

## 2. Design

### 2.1 New endpoints

```
POST /research/run
{
  "query": "What are the research gaps in retinal disease detection using computer vision?",
  "max_results_per_query": 10,
  "stop_after": "represent"        // query | collect | acquire | parse | sections | chunks | represent
}
→ 202 {"run_id": "...", "status": "RUNNING", "stage": "QUERY_UNDERSTANDING"}

GET  /research/run/{run_id}
→ {"run_id", "status", "stage", "error",
   "query_understanding": {...}, "queries": [...],
   "stages": {"collect": {"status","total","ok","failed"}, "acquire": {...}, ...}}

POST /research/run/{run_id}/resume      // restart from the failed/unfinished stage
POST /research/run/{run_id}/retry-downloads   // optional: DOWNLOAD_FAILED → NOT_CHECKED, rerun acquire→represent
```

The `run_id` is created at the start (it's also the query id), so you get it
immediately.

### 2.2 Orchestrator

New `app/orchestration/run_pipeline.py`, `RunPipeline.run(run_id)`:

1. **Query understanding**: `qu_service.run()` via `asyncio.to_thread`; save
   understanding + expanded queries on the run row.
2. **Collect**: searches the original question plus every expanded query
   (deduplicated, order kept), via `to_thread`.  Save failure or 0 papers
   marks the run FAILED with the reason, instead of hiding it.
3. **Acquire → parse → sections → chunks → represent**: calls the existing
   `run_acquisition`, `run_parsing`, `run_sectioning`, `run_chunking`,
   `run_representation` workers directly, in order, awaiting each.  No HTTP
   hops and no code duplicated.  A stage that processes 0 papers stops the run
   with a clear message (e.g. "no PDFs could be downloaded").
4. After each stage, writes stage, counts and timestamps to the run row.

Progress persists in the database, so `GET /research/run/{id}` survives a
server restart, and `/resume` continues from the stage that didn't finish
(the per-stage paper filters already make this safe).

A per-run lock (in-process `asyncio.Lock` plus a `status = RUNNING` check in
the DB) rejects a second start on the same run with 409.  Tasks are kept in a
set on `app.state` so they can't be garbage-collected.

### 2.3 Database (migration `09_run_pipeline_state.sql`)

```sql
ALTER TABLE research_runs
  ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'CREATED',
  ADD COLUMN IF NOT EXISTS stage TEXT,
  ADD COLUMN IF NOT EXISTS error TEXT,
  ADD COLUMN IF NOT EXISTS query_understanding JSONB,
  ADD COLUMN IF NOT EXISTS expanded_queries JSONB,
  ADD COLUMN IF NOT EXISTS stage_stats JSONB NOT NULL DEFAULT '{}'::jsonb,
  ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
```

Additive only: no deletes, existing runs keep working.

### 2.4 Retrieval stage

- `RetrieveRequest` gets `use_expanded_queries: bool = false`.  When true,
  retrieval runs for the original question plus the run's stored expanded
  queries and fuses them with RRF (the same logic as ResearchPipeline
  Stage 4, moved into `RetrievalService.aretrieve_multi`).  The per-paper cap
  applies after fusion.
- Exclude `REFERENCES` chunks (and other back matter: acknowledgements,
  author bios, page headers) from retrieval by default, with
  `include_back_matter` to override.
- Optional `section_types` filter, plus a preset `"gap_evidence"` covering
  Limitations, Future Work, Discussion, Conclusion and Results.
- Cache each run's loaded corpus in memory, refreshed when `/represent`
  finishes, so a request doesn't re-download every `chunks.json` (gap 18).
- Optional MMR re-ranking on top of the 3-per-paper cap.
- Clean up the remaining OTHER section types and page-header headings
  ("IJFMR…, Volume 8, Issue 2").

### 2.4b Run-scoped stage status (gap 17)

New table `run_papers(run_id, paper_id, acquire/parse/section/chunk/embed
status)`, backfilled from `paper_sources`.  Stages read and write status per
run.  A paper already processed for another run, whose files still exist, is
linked instead of reprocessed.  Every stage gets `force: true`, which replaces
the manual `reset_run_stages.py`.

### 2.5 Fixes to the existing endpoints (kept for manual use)

- `/collect` accepts an optional `run_id` (the one `/query` returned), runs
  the original question too, returns 502 when saving fails instead of 200,
  and runs off the event loop.
- `/query` runs off the event loop and creates the run row, so its id is the
  run id.
- Deduplication only looks at the papers of the run that was just collected.

---

## 3. Implementation steps

Each step ends with its tests passing; nothing is committed or pushed without
your say.

**Phase 1: link /query to /collect** (gaps 1–7, 16)
- Migration 09: the run-state columns on `research_runs` (2.3).
- `/query` creates the run row and saves the understanding and expanded queries; its id is the run_id.
- `/collect` accepts that `run_id` and searches the original question plus all stored expanded queries. Passing `query` and `expanded_queries` directly still works.
- `max_results_per_query` from the request is honoured, and the response says `capped: true` when the total cap is hit.
- A save failure returns an error, not a run_id that doesn't work.
- Collection runs off the event loop, and dedup only checks this run's papers.
- Tests: all queries searched, cap respected, save failure → error.

**Phase 2: run-scoped stages** (gap 17, part of 9)
- Migration 10: `run_papers` plus a backfill from `paper_sources`; nothing deleted.
- Stages read and write status per run.
- `force: true` on every stage.

**Phase 3: one-call pipeline** (gaps 8, 10, 11)
- `RunPipeline` orchestrator chaining the existing workers.
- `POST /research/run`, `GET /research/run/{id}` and `/resume`.
- Progress saved in the DB, a per-run lock, and a kept task set.
- Tests with fake workers: happy path, a stage with 0 papers, a stage raising, resume from the middle.

**Phase 4: retrieval stage** (gaps 12, 13, 18, 19)
- Multi-query fusion over the expanded queries.
- Back-matter exclusion, the `section_types` filter and the `gap_evidence` preset.
- Corpus cache, optional MMR, and the heading cleanup.

**Phase 5 (optional): acquisition recovery** (gap 15)
- Record a failure reason per paper.
- Retry with other open-access links already in the metadata (Unpaywall, arXiv, PMC).
- Summarise failures by reason.

After each phase: a real run on your laptop with the retinal question, reporting the run_id and per-stage counts. I'll show you each migration before running it. Docs (`api-test-guide.md`, `docs/retrieval_api.md`) are updated as we go.

Out of scope here: wiring gap detection (`/gaps`, gap 14) on top of this. That's the next plan.

---

## 4. Decisions for you

| Question | Options | My pick |
|----------|---------|---------|
| Where the pipeline stops by default | `represent` / continue into gap detection | **represent**, since gap detection isn't exposed yet |
| Where run progress is stored | DB columns on `research_runs` / in-memory JobStore | **DB**, so it survives restarts and can resume |
| Failed downloads | retry automatically once / only on `/retry-downloads` | **only on request**, because publishers often block repeated hits |
| Keep the separate stage endpoints | keep / remove | **keep**, for debugging single stages |
| Which phases now | 1–3 / 1–4 / all | **1–4**: Phase 4 is what makes retrieval good for gap detection |
| Search the original question too in `/collect` | yes / no | **yes** |
