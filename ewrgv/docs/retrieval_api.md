# Retrieval API Documentation

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/research/retrieve/dense` | POST | Dense (semantic) retrieval using vector similarity |
| `/api/v1/research/retrieve/sparse` | POST | Sparse (BM25) lexical keyword matching |
| `/api/v1/research/retrieve/knowledge-graph` | POST | Knowledge Graph concept/entity retrieval |
| `/api/v1/research/retrieve/hybrid` | POST | Hybrid fusion of all three methods via RRF |

---

## Request (all endpoints)

### Headers

```
Content-Type: application/json
```

### Body

| Field   | Type   | Required | Default | Constraints |
|---------|--------|----------|---------|-------------|
| `query` | string | **Yes**  | —       | Non-empty   |
| `top_k` | int    | No       | `10`    | 1–100       |

```json
{
  "query": "How do Transformer models use attention?",
  "top_k": 5
}
```

---

## Response (all endpoints)

```json
{
  "query": "How do Transformer models use attention?",
  "top_k": 5,
  "total_results": 5,
  "results": [
    {
      "chunk_id": "chunk-004-intro",
      "paper_id": "paper-004",
      "section": "Introduction",
      "text": "Self-attention-based architectures...",
      "score": 0.3946,
      "rank": 1,
      "retrieval_method": "dense",
      "metadata": {}
    }
  ]
}
```

### Result Fields

| Field              | Type   | Description |
|--------------------|--------|-------------|
| `chunk_id`         | string | Unique chunk identifier |
| `paper_id`         | string | Parent paper identifier |
| `section`          | string | Section heading |
| `text`             | string | Chunk text content |
| `score`            | float  | Method-specific score (higher is better) |
| `rank`             | int    | 1-based rank |
| `retrieval_method` | string | `"dense"`, `"sparse"`, `"knowledge_graph"`, or `"hybrid"` |
| `metadata`         | object | Method-specific provenance |

### Metadata by Endpoint

| Endpoint | `retrieval_method` | Metadata |
|----------|--------------------|----------|
| `/retrieve/dense` | `"dense"` | `{}` |
| `/retrieve/sparse` | `"sparse"` | `{}` |
| `/retrieve/knowledge-graph` | `"knowledge_graph"` | `matched_entities`, `relations` |
| `/retrieve/hybrid` | `"hybrid"` | `contributing_methods`, `rrf_k` |

---

## Error Responses

| Status | Condition |
|--------|-----------|
| 400    | Empty query, invalid top_k (≤0 or >100) |
| 422    | Missing required `query` field |

```json
{
  "error": "Query must be a non-empty string.",
  "detail": null
}
```

---

## Example Curl Requests

### Dense

```bash
curl -X POST http://localhost:8000/api/v1/research/retrieve/dense \
  -H "Content-Type: application/json" \
  -d '{"query": "Transformer attention mechanisms", "top_k": 3}'
```

### Sparse

```bash
curl -X POST http://localhost:8000/api/v1/research/retrieve/sparse \
  -H "Content-Type: application/json" \
  -d '{"query": "Transformer attention mechanisms", "top_k": 3}'
```

### Knowledge Graph

```bash
curl -X POST http://localhost:8000/api/v1/research/retrieve/knowledge-graph \
  -H "Content-Type: application/json" \
  -d '{"query": "Transformer attention mechanisms", "top_k": 3}'
```

### Hybrid

```bash
curl -X POST http://localhost:8000/api/v1/research/retrieve/hybrid \
  -H "Content-Type: application/json" \
  -d '{"query": "Transformer attention mechanisms", "top_k": 5}'
```

---

## How Each Method Works

### Dense (Cosine Similarity)

```
query → embed(query) → cosine_similarity(query_vec, chunk_vec) → top-K
```

Only the `DenseRetriever` is invoked. Uses `EmbeddingProvider` for vector embeddings.

### Sparse (BM25 Okapi)

```
query → tokenize(query) → BM25_score(query_tokens, chunk_tokens) → top-K
```

Only the `SparseRetriever` is invoked. Self-contained BM25 with k1=1.5, b=0.75.

### Knowledge Graph

```
query → find_entities(query) → traverse_1_hop() → collect_chunks() → score → top-K
```

Only the `KnowledgeGraphRetriever` is invoked. Scores: direct match = 1.0, related = 0.5, normalised to [0,1].

### Hybrid (Reciprocal Rank Fusion)

```
query → DenseRetriever → ranked list
      → SparseRetriever → ranked list
      → KGRetriever → ranked list
      → RRF(all lists, k=60) → deduplicated top-K
```

All three retrievers are invoked. Results fused via RRF: `score(d) = Σ 1/(k + rank_i(d))`.

---

## Complete Request Flow

```
POST /api/v1/research/retrieve/{method}
        │
        ▼
   RetrieveRequest (validated: non-empty query, 1 ≤ top_k ≤ 100)
        │
        ▼
   RetrievalService.retrieve(query, top_k, method)
        │
        ├── /dense  ──────► DenseRetriever only
        ├── /sparse ──────► SparseRetriever only
        ├── /knowledge-graph ► KGRetriever only
        └── /hybrid ──────► Dense + Sparse + KG → RRF fusion
        │
        ▼
   list[RetrievalResult]  (domain model)
        │
        ▼
   RetrieveResponse  (API schema)
```

---

## Query Understanding Compatibility

The `/query` endpoint returns `expanded_queries: list[str]`. The `/retrieve/*` endpoints accept raw query text.

**Usage:**
1. **Direct:** POST to any `/retrieve/*` endpoint with the original research question
2. **With expansion:** POST `/query` first, then POST to `/retrieve/*` once per expanded query
3. **Pipeline (future):** Stage 4 of ResearchPipeline will auto-feed expanded queries

No query_id or session state is required.
