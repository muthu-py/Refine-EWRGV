# API test guide (branch `integrate/gap-detection`)

Base URL: `http://localhost:8000/api/v1` (start with `uvicorn app.main:app --reload` from `ewrgv/`).
Swagger UI at `http://localhost:8000/docs` accepts the same bodies.

Test in this order: each step needs the `run_id` from `/collect`, and each
background job needs the previous one finished.  Poll the `.../status/{job_id}`
endpoint until `status` is `COMPLETED` (or `FAILED`) before moving on.

Keys needed in `.env`:
- `/query`: `LLM_API_KEY` (without it you get `status: "accepted"` and no expansion)
- `/collect` onwards: `SUPABASE_DATABASE_URL`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`
- `/represent`, `/retrieve/dense`, `/retrieve/hybrid` (dense part): `EMBEDDING_API_KEY`

---

## 0. Health

```
GET /health
```
No body.  Expect `200` with status ok.

---

## 1. Query understanding

```
POST /research/query
```
```json
{
  "query": "What are the research gaps in graph neural networks for CAN bus intrusion detection?",
  "max_papers": 20,
  "top_k_gaps": 5
}
```
Expect `status: "understood"`, an `understanding` object, and `expanded_queries`.
Copy `expanded_queries` into step 2.  Nothing is written to Supabase.

---

## 2. Collect literature

```
POST /research/collect
```
```json
{
  "query": "graph neural networks for CAN bus intrusion detection",
  "expanded_queries": [
    "in-vehicle network anomaly detection deep learning",
    "controller area network attack detection graph"
  ],
  "max_results_per_query": 10
}
```
Expect `status`, **`run_id`** (save it), `total_results`, `papers`.

Check in Supabase:
- `research_runs`: one row with this `run_id`.
- `papers`: new rows (all statuses `NOT_CHECKED` / not parsed).
- `paper_sources`: rows linking `run_id` to each paper.

⚠️ If saving fails, the endpoint still returns `200` with a `run_id`; the
error is only in the server log ("Failed to persist collection result").
If `research_runs` has no row, check the log before continuing.

---

## 3. Full-text acquisition (PDFs)

```
POST /research/acquire
```
```json
{ "run_id": "<run_id>" }
```
Returns a `job_id`.  Poll:
```
GET /research/<job_id>
```
Check: `papers.full_text_status` becomes `DOWNLOADED` (or `NO_OA_FOUND` /
`DOWNLOAD_FAILED` / `INVALID_DOCUMENT`); Storage bucket `papers` gets
`papers/<paper_id>/fulltext.pdf`.

---

## 4. Parse PDFs

```
POST /research/parse
```
```json
{ "run_id": "<run_id>" }
```
Poll `GET /research/parse/status/<job_id>`.
Check: `papers.parsing_status = PARSED`; Storage `papers/<paper_id>/parsed.json`.

---

## 5. Section extraction

```
POST /research/sections
```
```json
{ "run_id": "<run_id>" }
```
Poll `GET /research/sections/status/<job_id>`.
Check: `papers.section_status = EXTRACTED`; Storage `papers/<paper_id>/sections.json`
(look for sensible `section_type` values such as `LIMITATIONS`, `FUTURE_WORK`).

---

## 6. Chunking

```
POST /research/chunks
```
```json
{ "run_id": "<run_id>" }
```
Poll `GET /research/chunks/status/<job_id>`.
Check: `papers.chunk_status = CHUNKED`, `papers.chunks_storage_path` set;
Storage `papers/<paper_id>/chunks.json` (each chunk has `section_title`,
`section_type`, `page_start`, `page_end`).

---

## 7. Corpus health (any time after step 2)

```
GET /research/corpus/health?run_id=<run_id>
GET /research/corpus/health/inconsistencies?run_id=<run_id>
```
No body.  The first gives counts per stage; the second lists DB/Storage
mismatches (e.g. `CHUNKED` without a `chunks.json`).  Expect
`inconsistency_count: 0` after a clean run.

---

## 8. Representation (embeddings + BM25)

```
POST /research/represent
```
```json
{ "run_id": "<run_id>" }
```
Poll `GET /research/represent/status/<job_id>` (shows `embedded_chunks`,
`failed_chunks`, `embedding_status`, `bm25_status`).
Check: table `chunk_embeddings` has one row per chunk for
`embedding_model = text-embedding-3-small`; file
`data/artifacts/bm25/<run_id>/bm25_index.pkl` exists on the server machine.

---

## 9. Retrieval (needs step 6; dense needs step 8)

Same body for all three:
```json
{
  "run_id": "<run_id>",
  "query": "limitations of graph neural network intrusion detection on real vehicles",
  "top_k": 5
}
```
```
POST /research/retrieve/sparse
POST /research/retrieve/dense
POST /research/retrieve/hybrid
```
Expect each result to have `chunk_id`, `paper_id`, `section`, `section_type`,
`page_start`, `page_end`, `score`, `rank`, and `metadata` with `title`,
`publication_year`, `doi`.  Hybrid adds `metadata.contributing_methods`.

Things worth checking:
- `chunk_id`s match ids in that paper's `chunks.json`.
- Hybrid without `EMBEDDING_API_KEY` returns `retrieval_method: "sparse"` results.
- Error cases: unknown `run_id` → `404`; run with no chunks → `409`;
  `"query": ""` or `"top_k": 0` → `400`; missing `run_id` → `422`.

---

## 10. Placeholders (not implemented yet)

```
POST /research/gaps
```
```json
{ "job_id": "<any id>" }
```
```
POST /research/validate
```
```json
{ "job_id": "<any id>", "gap_ids": null }
```
Both return `status: "not_implemented"` with empty lists.  Gap detection
works in the pipeline code and tests but is not exposed through `/gaps` yet.

---

## 11. Knowledge graph and gap detection (new)

Full reference with real examples: `ewrgv/docs/gaps_kg_api.md` (and
`ewrgv/docs/api_test_guide.md` for every endpoint in order).  Needs a run
that has finished `/chunks` and `/represent`.

```
POST /research/kg/build          {"run_id": "<run_id>"}        → job_id
GET  /research/kg/status/<job_id>
GET  /research/kg/<run_id>?view=summary      (also: entities, statements, communities, papers, cytoscape, graphml)

POST /research/gaps              {"run_id": "<run_id>"}        → job_id
                                 add "include_llm_baseline": true to also run the LLM detector (1 LLM call)
GET  /research/gaps/status/<job_id>
GET  /research/gaps/<run_id>?view=summary    (also: full, compare; filter with &detector=D1 or D2)
```

Check in Supabase: `kg_graphs` and `gap_detection_results` each get one row
per run (migration 10).
