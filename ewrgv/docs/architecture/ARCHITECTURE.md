# EWRGV Architecture Document

## 1. Project Purpose

**Evidence-Weighted Research Gap Validation (EWRGV)** is an end-to-end RAG-based research gap identification system.

Its core research contribution is a dedicated validation process that separates:

```
Gap Detection  ≠  Gap Validation
```

The system first identifies *candidate* research gaps from a literature corpus, then runs the EWRGV validation pipeline to determine whether each candidate gap is genuinely unaddressed or already covered by existing work.

Each validated gap is classified as:

| Classification | Meaning |
|---|---|
| `VALID_GAP` | Strong evidence the gap is genuinely unaddressed |
| `UNCERTAIN_GAP` | Mixed or insufficient evidence; gap may or may not exist |
| `UNSUPPORTED_GAP` | Counter-evidence suggests the gap is already addressed |

---

## 2. Architecture Style

**Modular Monolith**

The system runs as a single application, but its internal modules have explicit responsibilities and interfaces such that individual components can later be replaced or extracted.

Explicitly NOT used:
- Microservices
- Message brokers / event queues
- Distributed processing
- Knowledge graphs / GNN / GraphRAG
- Fine-tuning

---

## 3. Dependency Direction

```
API Layer
    ↓
Orchestration Layer       (app/orchestration/)
    ↓
Application Modules       (gap_detection/, validation/, evidence/, coverage/, scoring/, explanation/)
    ↓
Domain Interfaces         (app/domain/interfaces.py)
    ↓
Infrastructure Implementations   (app/providers/, app/storage/)
```

**Rules:**
- API handlers do NOT import from providers or storage.
- Domain models do NOT import from infrastructure.
- Interfaces break the dependency between application logic and infrastructure.

---

## 4. Major Modules

| Package | Responsibility |
|---|---|
| `app/core/` | Config, logging, exceptions — no business logic |
| `app/domain/` | Domain models, enums, interfaces |
| `app/ingestion/` | Literature collection, parsing, chunking |
| `app/retrieval/` | Semantic, BM25, hybrid retrieval, reranking |
| `app/gap_detection/` | Candidate gap detection and ranking |
| `app/validation/` | **EWRGV validation pipeline** |
| `app/evidence/` | Evidence classification, aggregation, provenance |
| `app/coverage/` | Multi-dimensional coverage analysis |
| `app/scoring/` | Confidence scoring, gap classification |
| `app/explanation/` | Evidence-chain explanation generation |
| `app/providers/` | LLM, embedding, search provider adapters |
| `app/storage/` | Vector store, relational DB, repository adapters |
| `app/orchestration/` | End-to-end pipeline coordination |
| `app/api/` | FastAPI routes (thin layer) |

---

## 5. Module Responsibilities

### `app/core/`
- `config.py` — All settings via `pydantic-settings`; reads from `.env`
- `logging.py` — Centralised `get_logger()`, one-time `configure_logging()`
- `exceptions.py` — `EWRGVError` hierarchy; API exceptions carry `status_code`

### `app/domain/`
- `enums.py` — `GapType`, `EvidenceType`, `GapClassification`, `PipelineStage`, `QueryType`
- `models/` — Pydantic domain models: `Paper`, `DocumentChunk`, `ResearchQuery`, `CandidateGap`, `Evidence`, `ValidationResult`, `CoverageAssessment`
- `interfaces.py` — All `Protocol` and `ABC` interfaces

### `app/ingestion/`
- `collectors/` — `LiteratureCollector` implementations
- `parsers/` — `DocumentParser` implementations
- `sectioning/` — Section extraction
- `chunking/` — Text chunking strategies

### `app/retrieval/`
- `semantic.py` — Dense vector retrieval (`EmbeddingProvider` + `VectorStore`)
- `bm25.py` — Sparse keyword retrieval
- `fusion.py` — Reciprocal Rank Fusion
- `reranking.py` — Cross-encoder reranking

### `app/gap_detection/`
- `detector.py` — LLM-based candidate gap extraction
- `ranking.py` — Candidate ranking for Top-K selection
- `representation.py` — Structured gap decomposition

### `app/validation/` — **EWRGV Core**
- `validator.py` — `EWRGVValidator` — top-level EWRGV orchestrator
- `query_generation.py` — Verification query generation
- `direct_search.py` — Direct evidence search
- `terminology_search.py` — Synonym/alternative-term search
- `method_search.py` — Method-variation search
- `counter_search.py` — **Mandatory** counter-evidence search
- `iteration.py` — Iterative validation loop controller

### `app/evidence/`
- `classifier.py` — `LLMEvidenceClassifier`
- `aggregator.py` — `EvidenceAggregator` (de-duplication, weighting)
- `provenance.py` — Provenance tracking
- `strength.py` — Evidence strength assessment

### `app/coverage/`
- `analyzer.py` — `CoverageAnalyzer` (composes all dimensions)
- `retrieval.py`, `concepts.py`, `terminology.py`, `methods.py`, `corpus.py` — Dimension analyzers

### `app/scoring/`
- `confidence.py` — `EWRGVConfidenceScorer` (formula TBD)
- `classification.py` — `classify_by_confidence()` — pure mapping function

### `app/explanation/`
- `generator.py` — `LLMExplanationGenerator`
- `evidence_chain.py` — `build_evidence_chain()`

---

## 6. Data Flow

```
ResearchQuery
      ↓
[Query Expansion]
      ↓
[Literature Collection] ──→ Paper[]
      ↓
[Document Parsing + Chunking] ──→ DocumentChunk[]
      ↓
[Hybrid Retrieval: Semantic + BM25 + RRF] ──→ DocumentChunk[]
      ↓
[Reranking] ──→ DocumentChunk[] (top-n)
      ↓
[Gap Detection] ──→ CandidateGap[] (unvalidated)
      ↓
[Candidate Ranking] ──→ CandidateGap[] (top-k)
      ↓
─────────────── EWRGV BOUNDARY ───────────────
      ↓
[Generate Verification Queries]
      ↓
[Run Validation Searches] (direct + terminology + method + counter)
      ↓
[Classify Evidence] ──→ Evidence[]
      ↓
[Analyze Coverage] ──→ CoverageAssessment
      ↓
[Aggregate Evidence]
      ↓
[Score Confidence] ──→ float
      ↓
[Classify Gap] ──→ GapClassification
      ↓
[Generate Explanation] ──→ str
      ↓
ValidationResult
```

---

## 7. Why EWRGV is Separated from Gap Detection

This is the central architectural decision.

Gap detection produces **candidate gaps** — hypotheses that certain research areas may be underexplored. These hypotheses are based on limited initial retrieval.

EWRGV validation then **stress-tests** each hypothesis through:
1. **Expanded evidence search** — goes beyond the initial retrieval using targeted queries
2. **Terminology expansion** — finds evidence using alternative terms the gap detector may have missed
3. **Method variation search** — finds related work using different methodologies
4. **Mandatory counter-evidence search** — actively seeks papers that solve the gap

Without this separation, a naïve system would simply output whatever gaps the LLM identifies in the initial retrieval without verification. EWRGV transforms gap identification from a single-pass LLM prompt into a multi-stage evidence-weighted validation process.

The `validation/` package is **the research contribution**. It must never be merged into `gap_detection/`.

---

## 8. What Is Intentionally Not Implemented Yet

| Component | Status | Rationale |
|---|---|---|
| LLM calls (all providers) | Skeleton | Awaiting provider integration phase |
| Embedding computation | Skeleton | Awaiting embedding provider integration |
| Vector store indexing | Skeleton | Awaiting vector store integration |
| BM25 index | Skeleton | Awaiting retrieval implementation phase |
| Gap detection prompt | Skeleton | Awaiting gap detection implementation |
| EWRGV confidence formula | Interface only | Formula design is a research task |
| Coverage scoring formulas | Interface only | Design TBD after evidence collection works |
| Literature collectors | Skeleton | Awaiting API integration |
| Frontend | Not started | API-first; React frontend planned later |
| Evaluation datasets | Not started | Requires running pipeline first |
| Database schema | Not started | SQLAlchemy models deferred |
