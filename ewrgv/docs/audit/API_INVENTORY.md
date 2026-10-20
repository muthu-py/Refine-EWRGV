# API inventory

Prefix for all routes: `settings.API_V1_PREFIX`, default `/api/v1`. FastAPI docs are `/docs` and `/redoc`; UI mounts at `/` if `app/ui/` exists. Source schemas are Pydantic classes in `app/api/routes/research.py` and `gaps_kg.py`. This inventory reflects the current checkout (which contains a user modification to paper sorting in `research.py`). No live OpenAPI document or endpoint was requested/executed.

## Health and query/collection

| Method and route | Purpose / request | Response and execution | Status source |
|---|---|---|---|
| `GET /health` | Liveness/basic app health; no body | `HealthResponse`; synchronous | `routes/health.py` |
| `POST /research/query` | `QueryRequest {query, max_papers=20, top_k_gaps=5}` | `QueryResponse {job_id, query_id, status, message, query_type, understanding, expanded_queries, saved}`. LLM understanding when configured; DB save may fail/fallback. Not a background pipeline job. | no status route; `query_runs` when saved |
| `POST /research/collect` | `CollectRequest {query_id?, query?, expanded_queries=[], include_original_query=true, max_results_per_query?, max_total_results?, max_queries?}` | `CollectResponse {status,run_id,query_id,capped,queries_searched,providers_used,total_results,papers[],provider_errors[]}`. Search/persist performed in request path. | no collection job route |
| `GET /research/runs?limit=30` | List recent runs | Object with run metadata/counts | none |
| `GET /research/papers/{run_id}?q=&limit=50&offset=0&sort_by=downloaded` | Search/page run papers; sort values `downloaded`, `year_desc`, `year_asc`, `title_asc`, `title_desc` (working-tree code) | `{run_id,total,limit,offset,papers[]}` | none |

## Acquisition, preparation, retrieval

| Method and route | Purpose / request | Response and execution | Status route |
|---|---|---|---|
| `POST /research/acquire` | `{run_id}` | `202`-style accepted `AcquireResponse {job_id,run_id,status,message}`; BackgroundTasks worker | `GET /research/{job_id}` |
| `GET /research/{job_id}` | acquisition status | `JobStatusResponse`; job store only | — |
| `POST /research/parse` | `{run_id}` | `ParseResponse {job_id,run_id,status,message}`; BackgroundTasks | `GET /research/parse/status/{job_id}` |
| `POST /research/sections` | `{run_id}` | `SectionResponse {job_id,run_id,status,message}`; BackgroundTasks | `GET /research/sections/status/{job_id}` |
| `POST /research/chunks` | `{run_id}` | `ChunkResponse {job_id,run_id,status,message}`; BackgroundTasks | `GET /research/chunks/status/{job_id}` |
| `POST /research/represent` | `RepresentationStartRequest` (run ID and representation options; see route model) | `RepresentationStartResponse`; embedding/index worker in BackgroundTasks | `GET /research/represent/status/{job_id}` |
| `POST /research/retrieve/dense` | `RetrieveRequest`: `run_id,query,top_k=10` plus `section_types?,paper_ids?,year_from?,year_to?,include_noise=false` | `RetrieveResponse`; CrossEncoder reranking required; fail-closed with 503 if unavailable/disabled | no job status |
| `POST /research/retrieve/sparse` | same | BM25 candidates followed by required CrossEncoder reranking; fail-closed with 503 | no job status |
| `POST /research/retrieve/hybrid` | same | dense/sparse fused candidates followed by required CrossEncoder reranking; fail-closed with 503 | no job status |
| `POST /research/retrieve/multi` | `MultiRetrieveRequest`: `run_id`, explicit queries or `query_id`, top-k/filter fields | multi-query fused candidates followed by required CrossEncoder reranking; fail-closed with 503 | no job status |
| `GET /research/corpus/health?run_id=` | Corpus health summary | `CorpusHealthSummaryResponse` | none |
| `GET /research/corpus/inconsistencies?...` | Processing-stage consistency report, pagination/filter query params | `CorpusHealthReportResponse` | none |

## Knowledge graph, candidate gaps and validation

| Method and route | Purpose / request | Response and execution | Status route |
|---|---|---|---|
| `POST /research/kg/build` | `KGBuildRequest {run_id,query_id?,relevance_threshold?}` | `202 JobStarted`; uses `asyncio.create_task` for graph worker | `GET /research/kg/status/{job_id}` |
| `GET /research/kg/status/{job_id}` | Graph job state | `AnalysisJobStatus {job_id,run_id,kind,status,progress,summary,error,...}` | — |
| `GET /research/kg/{run_id}?view=&entity_type=&limit=` | Read stored KG; views include summary/entities/statements/claims/communities/papers/cytoscape/graphml | JSON or GraphML response; 404 if absent | none |
| `POST /research/gaps` | `GapDetectRequest {run_id,query_id?,detectors=[explicit,combination,temporal,topics],llm_wording=false,include_llm_baseline=false,rebuild_kg=false}`; `true` for either LLM flag is rejected (422) | `202 JobStarted`; non-LLM signal detectors use bounded, required CrossEncoder-reranked query/expansion context for grounding; reranker disabled/unavailable fails the job | `GET /research/gaps/status/{job_id}` |
| `GET /research/gaps/status/{job_id}` | Candidate job state | `AnalysisJobStatus` | — |
| `GET /research/gaps/{run_id}?view=&top=&detector=` | Stored candidates (summary/full/compare views) | Candidate result JSON; 404 if absent | none |
| `POST /research/validate` | `ValidateGapsRequest {run_id,gap_ids?,sources=[corpus,openalex,semantic_scholar],per_query=10,max_iterations?}`; sources must not be empty | `202 JobStarted`; validation worker via `asyncio.create_task` | `GET /research/validate/status/{job_id}` |
| `GET /research/validate/status/{job_id}` | Validation job state | `AnalysisJobStatus` | — |
| `GET /research/validate/{run_id}` | Read stored validation results | Persisted validation JSON; returns 404 when no validated row/results | none |

## Contract and reliability issues

1. **Job status durability:** status routes read in-memory `JobStore`; job IDs become unknown after process restart and are not shared across workers.
2. **Stage-specific status types:** acquisition/parsing/sectioning/chunking/representation and analysis use separate response models and status URLs. UI must implement each shape or normalize them.
3. **No single run state contract:** `GET /runs` describes collection runs, while processing status is per job and per paper. There is no durable aggregate indicating a run is ready for detection/validation.
4. **Query saved flag:** query route may return `saved=false`; clients must not assume `query_id` can always be reused by `/collect`.
5. **Collection synchronous contract:** provider latency/rate limiting can affect request duration; provider errors can coexist with partial result success.
6. **Validation completion can be false-positive:** current `KnowledgeGraphRepository.save_validations()` executes an `UPDATE ... WHERE run_id=$1` but does not inspect the asyncpg command tag/affected row count. `run_gap_validation()` can therefore proceed to completion even if no detection row matched. This matches a historic user report of candidate rows missing validation fields, though the live DB was not inspected.
7. **Working-tree API changes:** `GET /research/papers/{run_id}` accepts `sort_by` in current modified file; route and repository diff are uncommitted. That feature is not available in HEAD until committed.
8. **Validation bounds/enum detail:** `gaps_kg.py` Pydantic models are the contract source; generated docs should be treated as authoritative once code is stabilized. No external OpenAPI JSON or live API was inspected.

## UI integration readiness

The current UI README describes same-origin endpoints, run history and polling. The collection → processing steps are API-accessible, but clients must account for in-memory job state, partial provider results, missing persisted validation, 404 on absent results, and runs that may lack `query_id`. Stabilize versioned job/run status models and document readiness/prerequisite errors before relying on the UI for unattended workflows.
