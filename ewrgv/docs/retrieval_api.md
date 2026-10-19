# Retrieval API Documentation

Retrieval runs over the processed corpus of one research run: the
full-text `PaperChunk`s that `/chunks` wrote to `papers/{paper_id}/chunks.json`,
the dense embeddings `/represent` stored in `chunk_embeddings`, and the BM25
index `/represent` persisted at `BM25_INDEX_PATH/<run_id>/bm25_index.pkl`.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/research/retrieve/dense` | POST | Dense (semantic) retrieval over stored embeddings |
| `/api/v1/research/retrieve/sparse` | POST | Sparse (BM25) lexical keyword matching |
| `/api/v1/research/retrieve/hybrid` | POST | Dense + BM25 fused with Reciprocal Rank Fusion |
| `/api/v1/research/retrieve/multi` | POST | Several queries (e.g. the question + all its expansions) fused with RRF |

Applied to every endpoint:

- **Noise excluded by default.** REFERENCES / ACKNOWLEDGMENTS chunks and chunks whose text is mostly citations
  or publisher boilerplate (licence lines, journal headers) are never returned unless `include_noise: true`
  (`app/retrieval/filters.py`; on run `e0b54bcc` this excludes 143 of 943 chunks).
- **At most 3 hits per paper** (`RETRIEVAL_MAX_PER_PAPER`, 0 = off); if too few papers exist the best skipped
  hits backfill the list.
- **Corpus cache.** The run corpus is cached for `RETRIEVAL_CORPUS_CACHE_TTL` seconds (default 900) and cleared
  when `/chunks` or `/represent` finishes; repeated calls take ~0.6 s instead of re-downloading everything.

---

## Request (all endpoints)

| Field    | Type   | Required | Default | Constraints |
|----------|--------|----------|---------|-------------|
| `run_id` | string | **Yes**  | —       | An existing research run with chunked papers |
| `query`  | string | **Yes**  | —       | Non-empty   |
| `top_k`  | int    | No       | `10`    | 1–100       |
| `section_types` | list[string] | No | `[]` (all) | e.g. `["LIMITATIONS"]`; preset `"gap_evidence"` = LIMITATIONS, FUTURE_WORK, DISCUSSION, CONCLUSION, RESULTS |
| `paper_ids` | list[string] | No | `[]` (all) | Restrict to these papers |
| `year_from`, `year_to` | int | No | — | Inclusive publication-year bounds (results without a year are dropped when set) |
| `include_noise` | bool | No | `false` | Also search references / acknowledgments / boilerplate |

```json
{
  "run_id": "3f0c...",
  "query": "How do Transformer models use attention?",
  "top_k": 5
}
```

---

## Response (all endpoints)

```json
{
  "run_id": "3f0c...",
  "query": "How do Transformer models use attention?",
  "top_k": 5,
  "total_results": 5,
  "results": [
    {
      "chunk_id": "6b1e...",
      "paper_id": "a91d...",
      "section": "5 Limitations",
      "section_type": "LIMITATIONS",
      "page_start": 7,
      "page_end": 8,
      "text": "Our evaluation covers only English benchmarks...",
      "score": 0.3946,
      "rank": 1,
      "retrieval_method": "dense",
      "metadata": {"title": "...", "publication_year": 2021, "doi": "10...."}
    }
  ]
}
```

### Result Fields

| Field              | Type   | Description |
|--------------------|--------|-------------|
| `chunk_id`         | string | PaperChunk ID (from `chunks.json`) |
| `paper_id`         | string | Parent paper ID |
| `section`          | string | Original section heading (`section_title`) |
| `section_type`     | string | Classified `SectionType` (e.g. `LIMITATIONS`, `FUTURE_WORK`) |
| `page_start` / `page_end` | int | 1-based PDF pages the chunk spans |
| `text`             | string | Original chunk text |
| `score`            | float  | Method-specific score (higher is better) |
| `rank`             | int    | 1-based rank |
| `retrieval_method` | string | `"dense"`, `"sparse"` or `"hybrid"` |
| `metadata`         | object | Paper `title`, `publication_year`, `doi`; hybrid adds `contributing_methods`, `rrf_k` |

---

## Error Responses

| Status | Condition |
|--------|-----------|
| 400    | Blank `run_id` or query, invalid `top_k` (≤0 or >100), `year_from > year_to`, `/dense` without a configured embedding provider |
| 404    | `run_id` not found |
| 409    | The run has no chunked papers yet (run `/chunks`, and `/represent` for dense) |
| 422    | Missing `run_id` or `query` |
| 429    | Embedding API rate limit (API providers only); `Retry-After` header set |
| 502    | Embedding API rejected the key or the request (API providers only) |

---

## Multi-query retrieval: `POST /retrieve/multi`

| Field | Default | Meaning |
|---|---|---|
| `run_id` | — | Required |
| `query_id` | — | Stored `/query` result: uses the question + all expanded queries |
| `queries` | `[]` | Explicit queries (used instead of / without `query_id`) |
| `method` | `hybrid` | `dense`, `sparse` or `hybrid` per query |
| `top_k` | `10` | 1–100 |
| filters | — | Same `section_types`, `paper_ids`, `year_from`, `year_to`, `include_noise` |

Each query is ranked over the whole (filtered) corpus, the ranked lists are fused with RRF (k = 60), then the
per-paper cap applies. Results carry `retrieval_method: "multi_<method>"` and
`metadata.matched_queries` / `metadata.query_ranks`. `query` in the response is the queries joined by ` | `.

```json
{"run_id": "e0b54bcc-...", "query_id": "57393ebc-...", "method": "hybrid", "top_k": 10, "section_types": ["gap_evidence"]}
```

Errors: 400 no queries / bad method / bad `top_k`; 404 unknown `query_id` or run.

---

## Example

```bash
curl -X POST http://localhost:8000/api/v1/research/retrieve/hybrid \
  -H "Content-Type: application/json" \
  -d '{"run_id": "<run_id>", "query": "Transformer attention mechanisms", "top_k": 5}'
```

`scripts/verify_retrieval.py <run_id> "query"` prints sparse, dense and
hybrid results for a run from the command line.

---

## How Each Method Works

### Dense (cosine similarity)

```
query → embed(query) → cosine(query_vec, stored chunk_vec) → top-K
```

Chunk vectors come from `chunk_embeddings` for the configured
`EMBEDDING_MODEL`; chunks without a stored vector are embedded once on
demand.  Requires `EMBEDDING_API_KEY`.

### Sparse (BM25 Okapi)

```
query → tokenize(query) → BM25(query_tokens, chunk_tokens) → top-K
```

Uses the run's persisted index; it is rebuilt from the chunks when missing,
unreadable, or stale (its chunk IDs no longer match `chunks.json`).
k1=1.5, b=0.75.

### Hybrid (Reciprocal Rank Fusion)

```
query → DenseRetriever  → ranked list ┐
      → SparseRetriever → ranked list ┴→ RRF(k=60) → deduplicated top-K
```

`score(d) = Σ 1/(k + rank_i(d))`.  Without `EMBEDDING_API_KEY`, `/hybrid`
falls back to BM25 alone.

---

## Request Flow

```
POST /api/v1/research/retrieve/{method}
        │
        ▼
   RetrieveRequest (validated: run_id, non-empty query, 1 ≤ top_k ≤ 100)
        │
        ▼
   RunCorpusLoader.load(run_id) → RunCorpus (PaperChunks + embeddings + BM25)
        │
        ▼
   RetrievalService(corpus, embedding_provider).aretrieve(query, top_k, method)
        │
        ▼
   list[RetrievalResult] → RetrieveResponse
```

The `/query` endpoint returns `expanded_queries`; post each one to
`/retrieve/*` with the same `run_id` to retrieve for them as well.
ResearchPipeline Stage 4 does this automatically.
