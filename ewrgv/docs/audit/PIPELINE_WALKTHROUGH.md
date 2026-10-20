# Pipeline walkthrough

## Data flow

```mermaid
sequenceDiagram
  participant Client as UI / caller
  participant API as FastAPI routes
  participant DB as Postgres + Supabase Storage
  participant Providers as Literature/LLM/Embedding providers
  participant Worker as BackgroundTasks + JobStore
  Client->>API: POST /research/query
  API->>Providers: understand and expand (if configured)
  API->>DB: save query_runs
  API-->>Client: query_id, expansions, saved flag
  Client->>API: POST /research/collect
  API->>Providers: OpenAlex / Semantic Scholar searches
  API->>DB: research_runs, papers, paper_sources
  API-->>Client: run_id and partial provider results
  Client->>API: POST /research/acquire {run_id}
  API->>Worker: schedule job; keep status in process
  Worker->>Providers: OA discovery/download
  Worker->>DB: update paper acquisition state; store object
  Client->>API: POST parse → sections → chunks → represent
  API->>Worker: each is a separate job
  Worker->>DB: stage status/paths and chunk embeddings
  Client->>API: POST /research/kg/build (requires chunks + embeddings)
  API->>DB: verify chunked and represented paper status
  API->>Worker: schedule KG job
  Worker->>DB: save graph for run
  Client->>API: POST /research/gaps (requires stored KG)
  API->>Worker: schedule signal detection job
  Worker->>DB: save candidate JSONB by run
  Client->>API: POST /validate (requires saved candidates)
  Worker->>Providers: external support/counter searches
  Worker->>DB: save validation results, stats, timestamp
  Client->>API: GET status/results endpoints
```

## Stage-by-stage audit

| Stage | Status and implementation | Inputs / outputs / database effects | Prerequisites, tests and failure modes |
|---|---|---|---|
| 1. Query understanding | **Partial** — `QueryUnderstandingService` and `/research/query`. Structured schema and prompt service are implemented; absent credentials can return an offline/stub path. | Question → query type, entities/concepts/methods, expanded searches; `query_runs` persistence. | First stage. Tests: `test_query_understanding_service.py`, schemas/prompts, credential isolation. No test run. LLM errors or DB save failure can change status/`saved`. |
| 2. Query expansion/breakdown | **Partial** — same service, capped by settings. `/collect` accepts stored query ID or raw fields. | Search variants, optional query ID. | Depends on query understanding. Without stored query, later operations need explicit query. Ensure `max_queries` and expansion response semantics align. |
| 3. Literature collection | **Partial** — `LiteratureCollectionService`, Semantic Scholar/OpenAlex provider adapters, HTTP client, route `/research/collect`. | Queries → normalized bibliographic records; creates `research_runs`, canonical papers and provenance. | Depends on expanded terms/provider config. Provider errors are carried alongside partial results; rate limits/timeouts and non-deterministic provider recall. Tests exist for collectors/service/integration. |
| 4. Deduplication and persistence | **Partial** — dedup service plus paper/query repositories. DOI/title normalization. | Provider hits → canonical papers/source associations. | Depends on collection. Concurrent runs/DOI conflicts and transactions need proof. Tests exist. |
| 5. OA discovery/full-text acquisition | **Partial** — `acquisition/discovery.py`, `fulltext.py`, worker; Unpaywall, provider URLs, Supabase object store. | Run papers → file object and full-text status/path; counters in `AcquisitionJob`. | Depends on collection. Only OA-available/licensed content can be downloaded; storage/network failures leave per-paper failures. Job is process-local. |
| 6. Parsing | **Partial** — PyMuPDF `PDFParser` and worker. | PDF object → extracted text/document artifact, pages, parsing status/path. | Depends on successful acquisition. Scanned PDF OCR is not evidenced. Malformed PDFs, object lookup errors, status column drift. |
| 7. Section extraction | **Partial** — heading detection, classification, normalization and extractor. | Parsed document → section records/artifact and `section_status`. | Depends on parsing. Headingless documents and noisy layouts may be unclassified; heuristics require quality validation. |
| 8. Chunking and corpus | **Partial** — configurable chunker and `RunCorpusLoader`. | Sections → chunks with IDs, section types, paper provenance; corpus loads chunks/papers for a run. | Depends on sections. Artifact-path consistency, reruns and stage prerequisites matter. Existing unit tests. |
| 9. Embedding and retrieval | **Partial** — OpenAI/SentenceTransformer providers, representation worker, pgvector repository, BM25, hybrid retrieval with fusion, filters and CrossEncoder reranking. Public retrieval/validation callers require reranking and fail closed. | Chunks → embedding vectors and BM25 artifact; query → reranked `PaperChunk` results. | Depends on chunking. Model/dimension mismatch risk; migration 08 reset needed; no ANN index in migrations. CrossEncoder loads lazily; model/runtime availability remains an operational dependency. |
| 10. Candidate identification | **Partial** — KG builder and `GapDetectionService` support signal candidates: explicit statements, sparse concept combinations, temporal trends, sparse topics, fusion/ranking. Public `/gaps` defaults to signals and grounds them using bounded required CrossEncoder-reranked query/expansion context. Opt-in LLM detector proposals are merged with provenance and cited retrieved chunks; opt-in wording refinement preserves the original statement. | Run corpus + query → KG and ranked, provenance-bearing candidates; optionally include LLM proposals/refinement; saves per-run JSONB. | Depends on embeddings, corpus, query record and CrossEncoder availability. API requires the stored KG before `/gaps`; incomplete corpus inputs and missing KG return `409`. LLM options require configured credentials. Inadequate corpus can produce false sparsity; grounding and detector evaluation remain necessary. |
| 11. Evidence-weighted validation | **Partial** — `EWRGVValidator`; direct, terminology, method, counter searches; evidence classifier/aggregator; coverage metrics; `EWRGVConfidenceScorer`; explanation chain. | Candidate IDs → validation evidence, coverage, confidence and `VALID_GAP`/`UNCERTAIN_GAP`/`UNSUPPORTED_GAP`; result stored under `gap_detection_results`. | Depends on saved candidate rows, corpus/retrieval/embedder and optional external services. API returns `409` before job creation when no candidate result exists. Source unavailability, low coverage, or persistence failure must not produce false completion. Tests exist; historical reports of missing persisted validation. |

## Scoring and interpretation

`CandidateEvidenceScorer` scores initial support from passage relevance, section quality, recency and paper independence (see `gap_detection/evidence_scoring.py`). The separate validation confidence scorer (`scoring/confidence.py`) starts with `2 × gap_score + supporting evidence`, subtracts weighted counter/partial/contradictory evidence, and reduces confidence for external coverage. It caps confidence at 0.39 when fewer than three relevant evidence items or retrieval coverage is below 0.25; low corpus adequacy caps at 0.69. Configured thresholds are 0.70 for `VALID_GAP` and 0.40 for `UNCERTAIN_GAP`. Thus low confidence/unsupported can mean insufficient evidence, not definitive proof that the gap is false.

## End-to-end status

All listed stages have some source implementation, but several are partial and no evidence establishes a successful full run in the current configured environment. The Pipeline UI now shows query and collection ahead of the eight processing jobs, gates each stage on its prerequisites, and clears downstream completion when an earlier job is rerun. The KG, gap, and validation API routes also reject missing persisted prerequisites before scheduling work. Earlier processing endpoints remain separate jobs, and there is no durable pipeline run state, global retry/resume endpoint, or service-level guarantee after restart. “End-to-end” in README describes the product goal, not verified runtime behavior.
