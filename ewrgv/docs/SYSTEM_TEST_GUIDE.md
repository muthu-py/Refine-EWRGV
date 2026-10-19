# EWRGV full system test guide

This guide exercises the API from a research question through collection, processing, retrieval, knowledge-graph construction, candidate gap detection, and Phase 4 validation. It uses PowerShell examples for Windows. All research routes are under `/api/v1/research`.

## 1. Prerequisites

1. Start PostgreSQL/Supabase and the configured document storage.
2. Configure `.env`. Literature collection needs at least one configured search provider. The embedding default is OpenAI and requires a valid embedding key. For local embeddings, install `python -m pip install -e ".[sentence-transformers]"`, set `EMBEDDING_PROVIDER=sentence-transformers`, and set `EMBEDDING_MODEL` to a compatible local model. The first model use may download it; after it is cached, embedding does not need a hosted embedding API. OpenAlex and Semantic Scholar validation require internet access; a Semantic Scholar key is optional.
3. Install the updated project dependencies and apply database migrations, including migration 11 for validation persistence and the external-search cache:

   ```powershell
   cd C:\Users\Muthu\Downloads\Refine-ResearchGap\ewrgv
   python -m pip install -e .
   python scripts/apply_migrations.py
   ```

4. Start the API:

   ```powershell
   python -m uvicorn app.main:app --reload --port 8000
   ```

5. Open `http://127.0.0.1:8000/docs` to inspect the live OpenAPI schema. The API has no bearer-token requirement configured in these routes.

Set the PowerShell base URL and a reusable POST helper:

```powershell
$Base = "http://127.0.0.1:8000/api/v1/research"
$Api = "http://127.0.0.1:8000/api/v1"
$Headers = @{ "Content-Type" = "application/json" }
function Post-Json($Path, $Body) {
    Invoke-RestMethod -Method Post -Uri "$Base$Path" -Headers $Headers `
        -Body ($Body | ConvertTo-Json -Depth 20)
}
```

Background jobs return `202 Accepted` with a `job_id`. Poll the status URL from that response until `status` is `COMPLETED` or `FAILED`. Processing is sequential: wait for each stage to finish before starting the next.

## 2. Health check

### `GET /api/v1/health`

No request body.

```powershell
Invoke-RestMethod "$Api/health"
```

Expect HTTP 200 with `status: "ok"`, the app name/version and a timestamp. This confirms the API process is responding; it does not prove that the database or external providers are healthy.

## 3. Create a query

### `POST /query`

```powershell
$Query = Post-Json "/query" @{
    query = "What are the research gaps in retinal disease detection using computer vision?"
    max_papers = 20
    top_k_gaps = 5
}
$Query | ConvertTo-Json -Depth 10
$QueryId = $Query.query_id
```

Expected response fields include `query_id`, `status`, `saved`, `understanding`, and `expanded_queries`.

- `status: "understood"`: configured query-understanding provider returned structured fields and search expansions.
- `status: "accepted"`: query was stored, but no LLM key/provider was configured; expansions and understanding may be empty.
- `status: "error"`: query understanding failed. Check the provider configuration and response message before continuing.
- Continue only when `saved` is `true`, because `/collect` will load the stored query by `query_id`.

## 4. Collect the papers

### `POST /collect`

Pass `query_id` to use the saved question and its expansions. The collection stage searches the original question plus expansions by default.

```powershell
$Collection = Post-Json "/collect" @{
    query_id = $QueryId
    include_original_query = $true
    max_results_per_query = 10
    max_total_results = 100
    max_queries = 4
}
$Collection | ConvertTo-Json -Depth 8
$RunId = $Collection.run_id
```

Expect `run_id`, `queries_searched`, `providers_used`, `total_results`, `papers`, `capped`, and `provider_errors`. Save the returned `run_id`; all subsequent processing calls use it. Provider errors can coexist with partial results. If `run_id` is empty or collection status is `error`, resolve provider/database errors before continuing. `capped: true` means either the result cap or configured query cap was reached. A provider that exhausts its 429 retries is paused for the remainder of that collection run; successful papers from other providers are still returned and the response is marked partial.

You can instead provide `query` and optional `expanded_queries` without a `query_id`. Do not send an empty body. Typical errors are 400 for no query or invalid limits, 404 for an unknown `query_id`, and 503 if the result could not be saved.

## 5. Acquire full text

### `POST /acquire`

```powershell
$Acquire = Post-Json "/acquire" @{ run_id = $RunId }
$Acquire
$AcquireStatus = Invoke-RestMethod "$Base/$($Acquire.job_id)"
```

Poll `GET /api/v1/research/{job_id}` (the acquisition status route). Expect counts for total papers, acquired, failed and no-open-access papers. A job can complete even when some PDFs are unavailable. Continue to parsing when the status is `COMPLETED`; if zero papers were acquired, check access/open-access availability and collection results.

## 6. Parse PDFs

### `POST /parse`

```powershell
$Parse = Post-Json "/parse" @{ run_id = $RunId }
$ParseStatus = Invoke-RestMethod "$Base/parse/status/$($Parse.job_id)"
```

Poll the status URL until it completes. Expect `parsed`, `failed`, and `skipped` counts in `result_summary`. Only downloaded PDFs are eligible. If `parsed` is zero, inspect acquisition status and the parse job error.

## 7. Extract sections

### `POST /sections`

```powershell
$Sections = Post-Json "/sections" @{ run_id = $RunId }
$SectionsStatus = Invoke-RestMethod "$Base/sections/status/$($Sections.job_id)"
```

Poll `/sections/status/{job_id}`. Expect the job to complete with extracted/failed counts. Only parsed papers are eligible; already-extracted papers are skipped. A zero count usually means parsing did not produce eligible papers.

## 8. Create chunks

### `POST /chunks`

```powershell
$Chunks = Post-Json "/chunks" @{ run_id = $RunId }
$ChunksStatus = Invoke-RestMethod "$Base/chunks/status/$($Chunks.job_id)"
```

Poll `/chunks/status/{job_id}`. Expect chunked/failed counts. Only papers with extracted sections are eligible. If no papers are chunked, check the section job first.

## 9. Create dense and sparse representations

### `POST /represent`

```powershell
$Representation = Post-Json "/represent" @{ run_id = $RunId }
$RepresentationStatus = Invoke-RestMethod "$Base/represent/status/$($Representation.job_id)"
```

Poll `/represent/status/{job_id}`. Check:

- `status` is `COMPLETED`.
- `embedded_chunks` is close to or equal to `total_chunks`; investigate nonzero `failed_chunks`.
- `bm25_status` is `INDEXED`.
- `error` is empty.

This stage requires the configured embedding provider. With `sentence-transformers`, first-use model download may take time and requires the model to be available locally or downloadable. With OpenAI, the API key must be valid and have embedding access.

## 10. Check corpus health

### `GET /corpus/health?run_id={run_id}`

```powershell
Invoke-RestMethod "$Base/corpus/health?run_id=$RunId"
```

Expect paper counts for downloaded, parsed, sections extracted and chunked states. The counts may not match collected papers exactly because not every result will have accessible full text.

### `GET /corpus/health/inconsistencies?run_id={run_id}`

```powershell
Invoke-RestMethod "$Base/corpus/health/inconsistencies?run_id=$RunId"
```

Expect `papers_checked`, `inconsistency_count`, and an `inconsistencies` list. An empty list is the clean result. This endpoint reports storage/database mismatches; it does not repair them.

## 11. Test retrieval APIs

All retrieval endpoints use the same filter fields: `section_types`, `paper_ids`, `year_from`, `year_to`, and `include_noise`. `top_k` must be 1–100. The `gap_evidence` section preset searches limitations, future work, discussion, conclusion and results sections.

### `POST /retrieve/sparse`

```powershell
$Sparse = Post-Json "/retrieve/sparse" @{
    run_id = $RunId
    query = "limitations of retinal disease detection models"
    top_k = 10
    section_types = @("gap_evidence")
}
$Sparse | ConvertTo-Json -Depth 8
```

Expect `total_results` and ranked chunks with `chunk_id`, paper metadata, section/page information, `score`, `rank`, and `retrieval_method: "sparse"`.

### `POST /retrieve/dense`

```powershell
$Dense = Post-Json "/retrieve/dense" @{
    run_id = $RunId
    query = "limitations of retinal disease detection models"
    top_k = 10
    section_types = @("gap_evidence")
}
```

Expect the same response shape with semantic ranking and `retrieval_method: "dense"`. Dense retrieval requires stored vectors and a working query-embedding provider.

### `POST /retrieve/hybrid`

```powershell
$Hybrid = Post-Json "/retrieve/hybrid" @{
    run_id = $RunId
    query = "limitations of retinal disease detection models"
    top_k = 10
    section_types = @("gap_evidence")
}
```

Expect a fused ranking (`retrieval_method: "hybrid"`) combining dense and BM25 results. When dense embedding fails, hybrid returns an embedding error (429 for rate limits, 502 for provider rejection); run sparse retrieval separately while resolving the embedding configuration.

### `POST /retrieve/multi`

Use a stored query ID to retrieve the original question and all expansions, or supply a `queries` array.

```powershell
$Multi = Post-Json "/retrieve/multi" @{
    run_id = $RunId
    query_id = $QueryId
    method = "hybrid"
    top_k = 10
    section_types = @("gap_evidence")
}
```

Expect one RRF-fused result list. Results should include `metadata.matched_queries` and query-rank details. A 400 means no usable query or invalid method/limit; 404 indicates an unknown run/query; 409 generally means chunking or representation has not finished.

## 12. Build and inspect the knowledge graph

### `POST /kg/build`

```powershell
$KgJob = Post-Json "/kg/build" @{ run_id = $RunId; query_id = $QueryId }
$KgStatus = Invoke-RestMethod "$Base/kg/status/$($KgJob.job_id)"
```

Poll `/kg/status/{job_id}` until `COMPLETED`. Expect graph statistics including papers, on/off-topic papers, tagged chunks, entities, co-occurrence edges, communities, gap statements and result claims. `FAILED` includes an `error` explaining missing embeddings, chunks, database state, or another build failure.

### `GET /kg/{run_id}` views

```powershell
Invoke-RestMethod "$Base/kg/$RunId"                                      # summary
Invoke-RestMethod "$Base/kg/$RunId?view=papers"                          # relevance decisions
Invoke-RestMethod "$Base/kg/$RunId?view=entities&entity_type=METHOD&limit=25"
Invoke-RestMethod "$Base/kg/$RunId?view=statements&limit=20"
Invoke-RestMethod "$Base/kg/$RunId?view=claims&limit=20"
Invoke-RestMethod "$Base/kg/$RunId?view=communities"
Invoke-RestMethod "$Base/kg/$RunId?view=cytoscape"
Invoke-WebRequest "$Base/kg/$RunId?view=graphml" -OutFile "$PWD\kg-$RunId.graphml"
```

Expect these views to agree with the summary counts. The papers view shows each paper’s relevance and on-topic flag. Claims are rule-extracted directional results, not a complete interpretation of every paper. `404` means the graph has not been built for this run.

## 13. Detect candidate gaps (Phases 1–3)

### `POST /gaps`

The default detector set runs D1 through D4. The KG is built automatically if it is missing or belongs to a different query ID.

```powershell
$GapJob = Post-Json "/gaps" @{
    run_id = $RunId
    query_id = $QueryId
    detectors = @("explicit", "combination", "temporal", "topics")
    llm_wording = $false
    include_llm_baseline = $false
    rebuild_kg = $false
}
$GapStatus = Invoke-RestMethod "$Base/gaps/status/$($GapJob.job_id)"
```

Allowed detector names are `explicit` (D1), `combination` (D2), `temporal` (D3), and `topics` (D4); their full internal names are accepted too. Poll until `COMPLETED`. Expect counts per detector, candidate counts, grounding status, topic-clustering method, corpus time span and available external trends. OpenAlex trend data may be absent if the source is unavailable; that should not stop detection.

Optional settings:

- `llm_wording: true`: asks the configured LLM to reword eligible candidates; only a grounded rewrite is retained.
- `include_llm_baseline: true`: also runs the legacy LLM detector for comparison and requires an LLM key.
- `rebuild_kg: true`: rebuilds the KG before detecting candidates.

### `GET /gaps/{run_id}` views

```powershell
Invoke-RestMethod "$Base/gaps/$RunId?top=10"
Invoke-RestMethod "$Base/gaps/$RunId?view=full&top=10"
Invoke-RestMethod "$Base/gaps/$RunId?view=compare&top=10"
Invoke-RestMethod "$Base/gaps/$RunId?detector=temporal_staleness&top=10"
Invoke-RestMethod "$Base/gaps/$RunId?detector=topic_density&top=10"
```

Expect ranked candidate cards with `gap_statement`, `gap_type`, `gap_score`, `score_breakdown`, detector signals, concepts, evidence passages, counter hints, and a grounding result. These are hypotheses, not externally validated research gaps. `view=compare` includes the LLM baseline only if it was requested when detection ran. A new gap-detection run replaces the saved candidates and invalidates older validation results for that run.

## 14. Validate candidates (Phase 4)

Before first use, confirm migration 11 has been applied. Validation defaults to the run corpus, OpenAlex and Semantic Scholar. It may take longer than detection because it searches multiple query variants and sources.

### `POST /validate`

Validate the top 10 saved candidates:

```powershell
$ValidationJob = Post-Json "/validate" @{
    run_id = $RunId
    gap_ids = $null
    sources = @("corpus", "openalex", "semantic_scholar")
    per_query = 10
    max_iterations = 2
}
$ValidationStatus = Invoke-RestMethod "$Base/validate/status/$($ValidationJob.job_id)"
```

To validate a specific candidate, get its `gap_id` from `/gaps/{run_id}?view=full` and pass it in `gap_ids`, for example `gap_ids = @("<gap-id>")`. `gap_ids` can be omitted or null for the top 10. `sources` can be any non-empty subset of `corpus`, `openalex`, and `semantic_scholar`; `per_query` must be 1–50 and `max_iterations` 1–5 or null.

Poll `/validate/status/{job_id}`. Expect `COMPLETED` with number validated, classification counts, unavailable sources and external API call/cache statistics. If an external source cannot respond, it should be listed as unavailable; validation should still complete with the available sources.

### `GET /validate/{run_id}`

```powershell
$ValidationResults = Invoke-RestMethod "$Base/validate/$RunId"
$ValidationResults.validation_results | Select-Object gap_rank, gap_statement, classification, confidence, reason
```

Each result should contain a classification (`VALID_GAP`, `UNCERTAIN_GAP`, or `UNSUPPORTED_GAP`), confidence, reason, coverage dimensions, searches/queries used, evidence and the stepwise evidence chain. Low evidence should lead to low confidence or an insufficient-evidence reason, rather than an unsupported claim of certainty. Results are persisted per run and replaced by the next successful validation job.

## 15. Quick negative checks

These intentionally invalid requests should fail cleanly:

```powershell
Post-Json "/collect" @{}                                      # 400: query or query_id required
Post-Json "/gaps" @{ run_id = $RunId; detectors = @("magic") } # 422: invalid detector
Post-Json "/validate" @{ run_id = $RunId; sources = @() }       # 422: source required
Invoke-RestMethod "$Base/kg/not-a-run"                         # 404: run/KG not found
```

Common outcomes:

| Symptom | Check |
|---|---|
| `saved: false` or collection has no `run_id` | Database connection and collection provider configuration |
| Parse/sections/chunks job completes with zero work | The preceding stage completed and its eligible count is nonzero |
| `/represent` fails or dense retrieval returns an embedding error | Provider selection, model availability, or embedding credentials |
| Sparse retrieval works but dense/hybrid does not | Representation status and stored vector coverage |
| KG/gap job says no chunks | Run `/chunks` and `/represent` successfully first |
| `/validate` returns 404/no candidate gaps | Build the KG and run `/gaps` for the same `run_id` first |
| Validation finishes with `unavailable_sources` | External connectivity/rate limit; rerun later or select only available sources |
| Validation storage errors mention columns/table | Apply migration 11 and restart the API |

## 16. Full endpoint checklist

| API | Method | Body/query | Successful expectation |
|---|---|---|---|
| `/api/v1/health` | GET | none | HTTP 200, `status: ok` |
| `/api/v1/research/query` | POST | `query`, optional `max_papers`, `top_k_gaps` | stored `query_id` and understanding or accepted fallback |
| `/api/v1/research/collect` | POST | `query_id` or `query`, optional limits/expansions | persisted `run_id`, paper/provider counts |
| `/api/v1/research/acquire` | POST | `run_id` | job ID; poll `/research/{job_id}` |
| `/api/v1/research/parse` | POST | `run_id` | job ID; poll `/parse/status/{job_id}` |
| `/api/v1/research/sections` | POST | `run_id` | job ID; poll `/sections/status/{job_id}` |
| `/api/v1/research/chunks` | POST | `run_id` | job ID; poll `/chunks/status/{job_id}` |
| `/api/v1/research/represent` | POST | `run_id` | job ID; poll `/represent/status/{job_id}` |
| `/api/v1/research/retrieve/{dense,sparse,hybrid}` | POST | `run_id`, `query`, optional filters | ranked chunk results |
| `/api/v1/research/retrieve/multi` | POST | `run_id`, `query_id` or `queries` | multi-query fused results |
| `/api/v1/research/corpus/health` | GET | optional `run_id` query | processing-stage counts |
| `/api/v1/research/corpus/health/inconsistencies` | GET | optional `run_id` query | artifact mismatch report |
| `/api/v1/research/kg/build` | POST | `run_id`, optional `query_id`/threshold | job ID; poll `/kg/status/{job_id}` |
| `/api/v1/research/kg/{run_id}` | GET | optional `view`, `entity_type`, `limit` | saved graph/view/export |
| `/api/v1/research/gaps` | POST | `run_id`, optional query/detectors | job ID; poll `/gaps/status/{job_id}` |
| `/api/v1/research/gaps/{run_id}` | GET | optional `view`, `top`, `detector` | saved candidate gaps |
| `/api/v1/research/validate` | POST | `run_id`, optional `gap_ids`, `sources` | job ID; poll `/validate/status/{job_id}` |
| `/api/v1/research/validate/{run_id}` | GET | none | saved validation results |

`/docs` is the live reference if an implementation/schema differs from this guide. This guide describes expected behavior; actual counts, candidates and classifications vary with provider responses and corpus contents.
