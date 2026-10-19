# Pre-retrieval and retrieval: what is actually implemented

Codebase: `Refine-ResearchGap/ewrgv`, read on 2026-10-17. Every claim below comes from reading the code. All 424 tests pass, but they mock the database, Supabase Storage and every external API, so no stage has been tested end to end here.

## At a glance

| # | Stage | API | Implemented? | Main code |
|---|---|---|---|---|
| 2.1 | Query understanding and expansion | `POST /research/query` | **Yes** | `query_understanding/service.py` |
| 2.2 | Literature collection | `POST /research/collect` | **Yes** | `ingestion/collectors/service.py`, `providers/search/` |
| 2.3 | Deduplication | runs inside `/collect` | **Yes** | `ingestion/deduplication/service.py` |
| 2.4 | Open-access discovery and PDF download | `POST /research/acquire`, `GET /research/{job_id}` | **Yes** | `ingestion/acquisition/` |
| 2.5 | PDF parsing | `POST /research/parse`, `GET /research/parse/status/{job_id}` | **Yes** | `ingestion/parsers/` |
| 2.6 | Section extraction | `POST /research/sections`, `GET .../sections/status/{job_id}` | **Yes** | `ingestion/sectioning/` |
| 2.7 | Chunking | `POST /research/chunks`, `GET .../chunks/status/{job_id}` | **Yes** | `ingestion/chunking/` |
| 2.8 | Corpus health | `GET /research/corpus/health`, `GET .../corpus/health/inconsistencies` | **Yes** | `storage/corpus_health.py` |
| 2.9 | Dense and sparse representation | `POST /research/represent`, `GET .../represent/status/{job_id}` | **Yes** | `ingestion/representation/`, `retrieval/bm25_index.py` |
| R1 | BM25 retrieval | none | **No**: the index is built, but no query path exists | `retrieval/bm25.py` |
| R2 | Semantic (vector) retrieval | none | **No** | `retrieval/semantic.py` |
| R3 | Hybrid fusion (RRF) | none | **No**: `reciprocal_rank_fusion` returns `[]` | `retrieval/fusion.py` |
| R4 | Reranking | none | **No**: it just truncates the list to `top_n` | `retrieval/reranking.py` |

In short, everything up to "embeddings and a BM25 index exist for a run" is built. Nothing can search them yet, and there is no retrieval endpoint.

## Pre-retrieval, stage by stage

### 2.1 Query understanding (`POST /research/query`)
- One structured LLM call (OpenRouter by default, `openai/gpt-4o-mini`, temperature 0). It returns the domain, problem, task, methods, concepts, entities, datasets, metrics, constraints, research intent, `terminology_variants`, `method_variants`, and 5 to 8 `expanded_queries`, capped by `QUERY_EXPANSION_MAX_QUERIES=8`.
- The output is checked against a Pydantic schema, and failures are raised as `ProviderError`.
- Without an API key, the endpoint returns a stub with `status="accepted"`.
- **Gap:** the result is never stored. `/collect` builds a new `ResearchQuery` with a new ID, so the `run_id` doesn't match the `/query` ID. The understanding (terminology and method variants) is lost, even though retrieval and validation will need it.

### 2.2 Literature collection (`POST /research/collect`)
- The caller passes `query` and `expanded_queries`. Each query is sent to Semantic Scholar and OpenAlex (sync `httpx`, with retry and backoff). Papers are normalized into `Paper` with provenance (provider, source query). Limits: 20 per query and 200 in total.
- Results are written to Supabase Postgres (`research_runs`, `papers`, `paper_sources`).
- **Gaps:**
  - The loop runs every query against Semantic Scholar before OpenAlex starts. With 8 queries × 20 results, Semantic Scholar takes 160 of the 200 slots, so OpenAlex is starved.
  - There is no relevance, year or type filtering.
  - If the database write fails, the error is only logged and the call still returns a `run_id`.
  - The call blocks the event loop.

### 2.3 Deduplication
- DOIs are normalized, with a fallback key of title + authors + year. Duplicates merge into one canonical paper.
- **Gap:** each run deduplicates the whole database, not just the current run.

### 2.4 Acquisition (`POST /research/acquire`)
- It uses the provider's open-access URL when there is one. Otherwise it checks Unpaywall by DOI (only if `UNPAYWALL_EMAIL` is set). It then downloads the PDF and uploads it to Supabase Storage at `papers/{paper_id}/fulltext.pdf`.
- **Gaps:**
  - It only accepts a `Content-Type` of exactly `application/pdf`, so PDFs served as `octet-stream` are rejected.
  - When the provider URL is a landing page, the paper is marked `INVALID_DOCUMENT` without trying Unpaywall as a fallback.
  - Papers with no PDF drop out of the corpus entirely. Their abstracts are never chunked or indexed, so retrieval will only see papers that have open-access PDFs.

### 2.5 Parsing (`POST /research/parse`)
- Uses PyMuPDF `page.get_text("text")` to get plain text per page, and writes it to `parsed.json` in Storage. Documents with under 100 characters are flagged as low-text.
- **Gaps:** there's no OCR, no handling for two-column layouts or reading order, no removal of headers and footers, and no re-joining of hyphenated words.

### 2.6 Sectioning (`POST /research/sections`)
- A heading detector scores each line on 10 signals (numbering, known vocabulary, case, length, and caption, equation and list filters). Each section is classified into a `SectionType` such as ABSTRACT, METHODS, LIMITATIONS, FUTURE_WORK or REFERENCES, and written to `sections.json`.
- This is solid and well tested.

### 2.7 Chunking (`POST /research/chunks`)
- Splits by section, then paragraph, then sentence. Targets 500 tokens (max 600) with 75 tokens of overlap, and never crosses a section boundary. Each `PaperChunk` carries `section_type`, its pages and its character offsets. Output goes to `chunks.json`.
- **Gaps that hurt retrieval:**
  - **REFERENCES, ACKNOWLEDGMENTS and APPENDIX sections are chunked and indexed too.** Reference lists match keyword queries strongly, so they will crowd out BM25 results.
  - Chunks don't carry the paper title or the section heading, so embeddings lose context.
  - PyMuPDF's plain-text output rarely contains blank lines, so the paragraph split usually falls back to the sentence split.

### 2.9 Representation (`POST /research/represent`)
- **Dense:** chunk text is embedded with OpenAI `text-embedding-3-small` (1536 dimensions, batches of 32, retries) and stored in the pgvector table `chunk_embeddings (chunk_id, paper_id, embedding, embedding_model)`. This is incremental, so already-embedded chunks are skipped.
- **Sparse:** a `BM25Okapi` index over all chunks in the run, pickled to local disk at `data/artifacts/bm25/{run_id}/bm25_index.pkl`.
- **Gaps that block retrieval:**
  - `chunk_embeddings` stores no chunk text or section type, and the BM25 pickle stores only `chunk_id`s. To return a result, the text has to be fetched by downloading each paper's `chunks.json` from Storage. There is no chunk table in Postgres.
  - There is no vector index (HNSW or IVFFlat). The migration explicitly leaves it for the retrieval phase.
  - The BM25 index is on local disk, while everything else is in Supabase. It is lost on redeploy and has to be rebuilt for each run.
  - The vector column is hard-coded to `vector(1536)`, while `EMBEDDING_DIMENSION` is configurable.
  - The `/represent` error path crashes with a `NameError` because `JobStatus` and `HTTPException` aren't imported.

## Retrieval (`app/retrieval/`)
- `bm25_index.py` is the only working part. It handles tokenization (NFC normalization, lowercase, keeping hyphenated terms, a small stopword list), building the index, `query()` returning `(chunk_id, score)`, and save/load. It is used only at build time.
- `bm25.py` (`BM25Retriever`), `semantic.py` (`SemanticRetriever`), `fusion.py` (`reciprocal_rank_fusion` and `HybridRetriever`) and `reranking.py` (`CrossEncoderReranker`) are all stubs. They log a warning and return `[]`, or the input truncated.
- **Model mismatch:** every retrieval class is typed on `DocumentChunk`, but the pipeline produces `PaperChunk`, and nothing converts between them. The domain `Retriever` interface is synchronous, while the database, Storage and embedding calls are all async.
- The config already has `TOP_K=20`, `BM25_K=20`, `SEMANTIC_K=20` and `RERANK_TOP_N=10`. `sentence-transformers`, which the cross-encoder needs, is not installed.
- There is no retrieval API endpoint.

## What the retrieval stage needs (proposed)
1. Store the query understanding against the `run_id`, so retrieval can use the expanded queries and terminology variants.
2. Exclude REFERENCES, ACKNOWLEDGMENTS and APPENDIX chunks from the indexes, or filter them at query time.
3. Put chunk text and metadata into Postgres (a `chunks` table, or columns on `chunk_embeddings`), so results don't need a Storage download, and add an HNSW index.
4. Implement BM25 and pgvector retrievers on `PaperChunk`, RRF fusion across retrievers and across expanded queries, and an optional cross-encoder rerank.
5. Add the APIs: `POST /research/retrieve` (query or expanded queries, mode `hybrid|semantic|bm25`, `top_k`, `rerank`, `section_types` filter), returning chunks with per-retriever scores, ranks and diagnostics. Add `GET /research/retrieve/readiness/{run_id}` to report embedded chunk counts and whether the BM25 index exists.
