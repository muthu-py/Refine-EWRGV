# Retrieval System — Phase 2.5 Documentation

## Architecture Overview

```
Research Query
    ↓
Query Understanding (Phase 2.1)
    ↓
 ┌──────────────┬──────────────┬─────────────────┐
 │ Dense        │ Sparse       │ Knowledge Graph  │
 │ Retrieval    │ Retrieval    │ Retrieval        │
 │ (cosine sim) │ (BM25)       │ (concept match)  │
 └──────────────┴──────────────┴──────────────────┘
    ↓                ↓                ↓
    └────────────────┴────────────────┘
                     ↓
        Reciprocal Rank Fusion (RRF)
                     ↓
        Unified Retrieval Results
                     ↓
        [Gap Detection → EWRGV Validation]
```

## Components

### 1. Corpus Storage Stub

**File:** `app/stubs/stub_corpus.py`
**Class:** `StubCorpusStore`
**Interface:** `CorpusStore` (defined in `app/domain/interfaces.py`)

**⚠️ Development stub — will be replaced by a real database-backed implementation.**

Provides a deterministic in-memory corpus of 5 academic ML/NLP papers
with 13 total document chunks.  Papers include:

| Paper ID    | Title                                                  | Chunks |
|-------------|--------------------------------------------------------|--------|
| paper-001   | Attention Is All You Need                              | 3      |
| paper-002   | BERT: Pre-training of Deep Bidirectional Transformers  | 2      |
| paper-003   | GPT-3: Language Models are Few-Shot Learners           | 3      |
| paper-004   | An Image is Worth 16x16 Words (ViT)                   | 2      |
| paper-005   | Retrieval-Augmented Generation (RAG)                   | 3      |

**Replacement path:**  Implement a class satisfying `CorpusStore` that reads
from the production database.  No retrieval code needs to change.

---

### 2. Text Representation Stub

**File:** `app/stubs/stub_text_representation.py`
**Classes:** `StubEmbeddingProvider`, `StubLexicalRepresenter`
**Interface:** `EmbeddingProvider` (defined in `app/domain/interfaces.py`)

**⚠️ Development stubs — will be replaced by real embedding models.**

#### Dense Embeddings (`StubEmbeddingProvider`)
- Produces deterministic 128-dimensional vectors using SHA-256 hashing
- Same input → same output (always)
- All vectors are L2-normalised to unit length
- No external API calls

#### Lexical Tokenisation (`StubLexicalRepresenter`)
- Lowercase → remove punctuation → split → remove stop words
- Deterministic: same input → same tokens
- No stemming/lemmatisation (deliberate simplification)

**Replacement path:**
- Dense: Replace with `SentenceTransformerEmbeddingProvider` or
  `OpenAIEmbeddingProvider` satisfying the `EmbeddingProvider` protocol.
- Lexical: Replace with a proper NLP tokeniser (spaCy, NLTK, etc.)

---

### 3. Knowledge Graph Stub

**File:** `app/stubs/stub_knowledge_graph.py`
**Class:** `StubKnowledgeGraphStore`
**Interface:** `KnowledgeGraphStore` (defined in `app/domain/interfaces.py`)

**⚠️ Development stub — will be replaced by a real KG database.**

Contains a hand-crafted graph of 12 ML/NLP entities with 18 directed
relationships:

```
Transformer ──uses──> Self-Attention
    ↑                     ↑
    │ based_on             │ uses
    │                     │
  BERT ──uses──> Pre-training ──precedes──> Fine-tuning
  GPT-3 ──uses──> Pre-training
  ViT ──based_on──> Transformer
  RAG ──uses──> Dense Passage Retrieval ──based_on──> BERT
```

Entity-to-chunk mappings link each concept to relevant document chunks
from the mock corpus.

**Replacement path:**  Implement a class satisfying `KnowledgeGraphStore`
backed by Neo4j, RDF, or a custom graph database.

---

### 4. Dense Retrieval

**File:** `app/retrieval/semantic.py`
**Class:** `DenseRetriever`

**Similarity metric:** Cosine Similarity

```
cos(a, b) = (a · b) / (‖a‖ × ‖b‖)
```

**Flow:**
1. Embed the query using the injected `EmbeddingProvider`
2. Embed all corpus chunks (lazy, cached after first call)
3. Compute cosine similarity between query embedding and each chunk
4. Sort by similarity descending (chunk_id tiebreaker)
5. Return top-K as `RetrievalResult` objects with `retrieval_method="dense"`

**Configuration:**
- `top_k`: number of results (default 20)

---

### 5. Sparse Retrieval (BM25)

**File:** `app/retrieval/bm25.py`
**Class:** `SparseRetriever`

**Scoring formula:** BM25 Okapi

```
score(q, d) = Σ IDF(t) × [tf(t,d) × (k1 + 1)] / [tf(t,d) + k1 × (1 - b + b × |d|/avgdl)]

IDF(t) = ln((N - n(t) + 0.5) / (n(t) + 0.5) + 1)
```

**Flow:**
1. Tokenize/normalise the query
2. Build BM25 index from corpus chunks (lazy, cached)
3. Score each document using the BM25 formula
4. Sort by score descending (chunk_id tiebreaker)
5. Return top-K as `RetrievalResult` objects with `retrieval_method="sparse"`

**Configuration:**
- `top_k`: number of results (default 20)
- `k1`: term frequency saturation (default 1.5)
- `b`: length normalisation (default 0.75)

Self-contained — no external library (rank_bm25, Elasticsearch) required.

---

### 6. Knowledge Graph Retrieval

**File:** `app/retrieval/knowledge_graph.py`
**Class:** `KnowledgeGraphRetriever`

**Scoring heuristic:**
- Direct entity match: +1.0 per matching entity
- Related entity (1-hop): +0.5 per relationship
- Scores normalised to [0, 1]

**Flow:**
1. Find entities/concepts matching the query in the KG
2. Collect chunk IDs from directly linked entities (score = 1.0)
3. Traverse 1-hop relationships to related entities
4. Collect chunk IDs from related entities (score = 0.5)
5. Aggregate scores per chunk, normalise
6. Sort by score descending (chunk_id tiebreaker)
7. Return top-K with `retrieval_method="knowledge_graph"` and
   metadata including `matched_entities` and `relations`

**Configuration:**
- `top_k`: number of results (default 20)

---

### 7. Hybrid/RRF Fusion

**File:** `app/retrieval/fusion.py`
**Class:** `HybridRetriever`

**Fusion formula:** Reciprocal Rank Fusion (RRF)

```
score(d) = Σ  1 / (k + rank_i(d))
```

Where:
- `k` = smoothing constant (default 60, configurable)
- `rank_i(d)` = 1-based rank of document `d` in ranked list `i`
- Sum is over all ranked lists containing document `d`

RRF is used because the three retrieval methods produce scores on
fundamentally different scales.  RRF normalises this by using rank
positions only.

**Flow:**
1. Run DenseRetriever, SparseRetriever, KnowledgeGraphRetriever
2. Collect all result lists
3. For each unique chunk_id, compute the RRF score
4. Deduplicate by chunk_id
5. Sort by RRF score descending (chunk_id tiebreaker)
6. Trim to final top_k
7. Return with `retrieval_method="hybrid"` and metadata including
   `contributing_methods` and `rrf_k`

**Configuration:**
- `top_k`: final result count (default 20)
- `rrf_k`: smoothing constant (default 60)
- `per_retriever_k`: results requested from each retriever (default 2× top_k)

**Reference:** Cormack, Clarke & Buettcher (2009). "Reciprocal Rank Fusion
outperforms Condorcet and individual Rank Learning Methods." SIGIR '09.

---

### 8. Unified Retrieval Result

**File:** `app/domain/models/retrieval_result.py`
**Class:** `RetrievalResult` (Pydantic model)

| Field              | Type   | Description                                    |
|--------------------|--------|------------------------------------------------|
| `chunk_id`         | str    | Unique chunk identifier                        |
| `paper_id`         | str    | Parent paper identifier                        |
| `section`          | str    | Section heading                                |
| `text`             | str    | Chunk text content                             |
| `score`            | float  | Retrieval score (method-specific)              |
| `rank`             | int    | 1-based rank in result list                    |
| `retrieval_method` | str    | `"dense"`, `"sparse"`, `"knowledge_graph"`, `"hybrid"` |
| `metadata`         | dict   | Optional: entities, relations, provenance, etc.|

---

## Replacing Stubs with Real Implementations

All retrieval code depends on **interfaces**, not on stubs directly.
To replace a stub:

1. Implement the corresponding protocol interface
2. Inject the new implementation where the stub was used
3. No retrieval logic changes needed

| Stub                       | Interface            | Future Implementation              |
|----------------------------|----------------------|------------------------------------|
| `StubCorpusStore`          | `CorpusStore`        | Database-backed corpus store       |
| `StubEmbeddingProvider`    | `EmbeddingProvider`  | OpenAI / Sentence-Transformers     |
| `StubLexicalRepresenter`   | *(direct injection)* | spaCy / NLTK tokeniser             |
| `StubKnowledgeGraphStore`  | `KnowledgeGraphStore`| Neo4j / RDF graph store            |
