# Retrieval System

## Architecture Overview

```
/acquire → /parse → /sections → /chunks ──→ papers/{paper_id}/chunks.json (PaperChunk[])
                                /represent ─→ chunk_embeddings (pgvector)
                                           └→ BM25_INDEX_PATH/<run_id>/bm25_index.pkl
                                   ↓
                     RunCorpusLoader.load(run_id) → RunCorpus
                                   ↓
              ┌────────────────────┴────────────────────┐
              │ Dense retrieval        Sparse retrieval │
              │ (cosine, stored        (BM25, persisted │
              │  embeddings)            index)          │
              └────────────────────┬────────────────────┘
                                   ↓
                     Reciprocal Rank Fusion (RRF, k=60)
                                   ↓
                RetrievalResult[] / canonical PaperChunk[]
                                   ↓
                 [Gap Detection → EWRGV Validation]
```

## Components

### 1. Chunk model: `PaperChunk`

**File:** `app/domain/models/paper_chunk.py`

The single chunk model used by retrieval, gap detection and validation.
Each chunk carries original paper text plus provenance:
`chunk_id → paper_id → section_id / section_title / section_type → page_start..page_end`.
Bibliographic data (title, year, DOI, authors) lives on `Paper` and is
looked up through the corpus.

### 2. Run corpus: `RunCorpus` / `RunCorpusLoader`

**File:** `app/ingestion/corpus/run_corpus.py`

`RunCorpus` is the in-memory `CorpusStore` for one research run:

| Method | Returns |
|--------|---------|
| `get_all_chunks()` / `get_chunks_by_paper_id(id)` / `get_chunk_by_id(id)` | PaperChunks |
| `get_all_papers()` / `get_paper_by_id(id)` | Papers |
| `get_embedding(chunk_id)` | stored vector or `None` |
| `bm25_index` | the run's `BM25Index` (built lazily if none was loaded) |
| `stats` | papers, papers with chunks, chunks, embedded chunks, chunks by section type |

`RunCorpusLoader(paper_repo, storage_client, embedding_repo=None, embedding_model=None).load(run_id)`:

1. `PaperRepository.get_papers_for_corpus(run_id)` — the run's papers with `chunk_status = CHUNKED`.
2. Downloads each paper's `chunks.json` (`chunks_storage_path`, or the default path).  A paper whose artifact cannot be read is logged and skipped.
3. With an `embedding_repo`, loads stored vectors for `EMBEDDING_MODEL` via `EmbeddingRepository.get_embeddings_for_papers`.
4. Loads the persisted BM25 index if present.  It is rebuilt from the chunks when unreadable or stale (its chunk IDs differ from the chunks just loaded, e.g. after re-chunking).

`RunCorpus` does no I/O, so tests build it directly from in-memory chunks
(`tests/fixtures/sample_corpus.py`).

### 3. Dense Retrieval

**File:** `app/retrieval/semantic.py` — **Class:** `DenseRetriever(embedding_provider, corpus_store)`

```
cos(a, b) = (a · b) / (‖a‖ × ‖b‖)
```

1. Use the stored vector of each chunk (`corpus.get_embedding`); embed chunks without one in a single `embed_batch` call (cached).
2. Embed the query.
3. Rank by cosine similarity (chunk_id tiebreaker); empty queries return nothing.

`retrieve()` returns the corpus's PaperChunks; `retrieve_results()` returns
`RetrievalResult`s with `retrieval_method="dense"`.

### 4. Sparse Retrieval (BM25)

**File:** `app/retrieval/bm25.py` — **Class:** `SparseRetriever(corpus_store, bm25_index=None)`

```
score(q, d) = Σ IDF(t) × [tf(t,d) × (k1 + 1)] / [tf(t,d) + k1 × (1 - b + b × |d|/avgdl)]
```

Uses `BM25Index` (`app/retrieval/bm25_index.py`, rank_bm25 Okapi,
k1=1.5, b=0.75): the explicit index if given, else the corpus's
`bm25_index`, else one built from the chunks.  Hits whose chunk is not in
the corpus are skipped.  `retrieval_method="sparse"`.

### 5. Hybrid/RRF Fusion

**File:** `app/retrieval/fusion.py` — **Class:** `HybridRetriever`

```
score(d) = Σ  1 / (k + rank_i(d))
```

- `k` = smoothing constant (default 60)
- `rank_i(d)` = 1-based rank of `d` in list `i`

Dense and BM25 scores are on different scales, so RRF fuses by rank only.
Results are deduplicated by chunk_id, sorted by RRF score (chunk_id
tiebreaker), and keep the section/page fields of the source result plus
`contributing_methods` and `rrf_k` in metadata.  An optional third ranked
retriever can be passed as `kg_retriever`.

Reference: Cormack, Clarke & Buettcher (2009), SIGIR '09.

### 6. RetrievalService

**File:** `app/retrieval/service.py` — `RetrievalService(corpus_store, embedding_provider=None, require_reranking=False)`

Dispatches `dense`, `sparse` or `hybrid`.  Without an embedding provider,
`dense` raises `BadRequestError` and `hybrid` falls back to BM25.
`aretrieve()` is the async entry point used by the API: it embeds the query
and any missing chunk texts with the provider's async API first, so the
sync retrievers never call the network inside the event loop.

Reranking is enabled by default. The service caps the candidate pool and scores the
leading query/passage pairs with a lazily loaded SentenceTransformers
CrossEncoder, then applies the per-paper cap and `top_k`. The original retrieval
score remains in `RetrievalResult.score`; raw reranker logits and reranking
status are recorded in metadata. API retrieval, validation and gap-detection
callers set `require_reranking=True`, so disabled or failed reranking is an
error rather than a silent fallback. The `/gaps` job also fetches bounded,
reranked query/expansion context before signal detection and uses it in the
textual-candidate grounding gate. Install runtime dependencies with
`pip install -e .` or `pip install -r requirements.txt`. See `retrieval_api.md`
for environment variables and failure policy.

### 7. Unified Retrieval Result

**File:** `app/domain/models/retrieval_result.py`

| Field              | Type   | Description |
|--------------------|--------|-------------|
| `chunk_id`         | str    | PaperChunk ID |
| `paper_id`         | str    | Parent paper ID |
| `section`          | str    | Section heading |
| `section_type`     | str    | `SectionType` value |
| `page_start` / `page_end` | int | PDF pages |
| `text`             | str    | Chunk text |
| `score`            | float  | Method-specific score |
| `rank`             | int    | 1-based rank |
| `retrieval_method` | str    | `"dense"`, `"sparse"`, `"hybrid"` |
| `metadata`         | dict   | Paper title/year/DOI; fusion provenance |

## Verifying against a real run

```bash
python scripts/verify_retrieval.py <run_id> "your query" 5
```
