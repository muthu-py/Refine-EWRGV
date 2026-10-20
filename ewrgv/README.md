# EWRGV — Evidence-Weighted Research Gap Validation

> An end-to-end RAG-based research gap identification and validation system.

---

## Project Purpose

EWRGV identifies and **validates** research gaps in academic literature using a retrieval-augmented pipeline.

The central research contribution is the **EWRGV validation process**, which separates gap detection from gap validation:

```
Gap Detection  ≠  Gap Validation
```

Each candidate gap is validated through multi-dimensional evidence search, classified as:

| Output | Meaning |
|---|---|
| `VALID_GAP` | Strong evidence the gap is genuinely unaddressed |
| `UNCERTAIN_GAP` | Mixed evidence; gap credibility is uncertain |
| `UNSUPPORTED_GAP` | Counter-evidence suggests the gap is already addressed |

---

## Current Status

> **Research MVP — v0.1.0**

The repository contains working API paths for literature collection and processing,
dense/sparse/hybrid retrieval, knowledge-graph construction, signal-based gap
identification, and evidence-weighted validation. These stages are started
individually and tracked as background jobs; the API does not yet provide a
durable, restart-safe end-to-end run coordinator.

| Component | Status |
|---|---|
| Domain models | ✅ Complete |
| Interfaces / Protocols | ✅ Complete |
| Orchestration | 🟡 Separate stage jobs; no durable run coordinator |
| EWRGV validation | 🟡 Implemented; depends on corpus and external sources |
| Retrieval API | ✅ Dense, sparse, hybrid, and multi-query endpoints |
| Cross-encoder reranking | ✅ Enabled by default for retrieval APIs |
| Candidate gap identification | ✅ Knowledge-graph signal detectors; `/gaps` is non-LLM |
| API | ✅ Health, research, knowledge-graph, gap, and validation routes |
| Config & logging | ✅ Complete |
| LLM / embedding providers | 🟡 Configurable; query understanding and embeddings may use providers |
| Literature collectors | ✅ Semantic Scholar and OpenAlex |
| Frontend | 🟡 API-served interface |

`POST /api/v1/research/gaps` generates candidates with explicit-statement,
combination-sparsity, temporal-staleness, and topic-density signals. It does
not call an LLM; requests that enable LLM wording or the LLM baseline are
rejected. Cross-encoder retrieval supplies ranked passages for grounding.

Reranking uses `cross-encoder/ms-marco-MiniLM-L-6-v2` by default and lazily
loads it on the first non-empty retrieval request. Install runtime dependencies
with `pip install -r requirements.txt`. If the model cannot load or score, API
retrieval fails clearly instead of returning unreranked results. Embedding and
query-understanding APIs may still use their configured providers; the `/gaps`
candidate detector itself is signal-based.

---

## Architecture Overview

**Style:** Modular monolith — single application, modular internals.

```
API Layer (FastAPI)
        ↓
Orchestration (ResearchPipeline)
        ↓
Application Modules
    ┌───────────────┬──────────────┬───────────────────┐
    ↓               ↓              ↓                   ↓
Ingestion      Retrieval    Gap Detection       EWRGV Validation
                                                    ↓
                                            Evidence / Coverage
                                                    ↓
                                           Scoring / Explanation
        ↓
Domain Interfaces (Protocols / ABCs)
        ↓
Infrastructure (providers/, storage/)
```

See [docs/architecture/ARCHITECTURE.md](docs/architecture/ARCHITECTURE.md) for full detail.

---

## Setup Instructions

### Prerequisites

- Python 3.11+
- pip or pip-tools

### 1. Clone and navigate

```bash
git clone <repo-url>
cd ewrgv
```

### 2. Create virtual environment

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env with your API keys and preferences
```

### 5. Run the API server

```bash
uvicorn app.main:app --reload
```

API docs available at: http://localhost:8000/docs

### 6. Run tests

```bash
pytest tests/ -v
```

---

## Development Priorities

- Evaluate retrieval recall and cross-encoder ranking on a fixed, labeled corpus.
- Measure signal-based candidate quality and grounding, including false positives
  from incomplete full-text coverage and sparse corpora.
- Exercise API jobs against a disposable PostgreSQL/Supabase database and verify
  persisted detection and validation results across restarts.
- Improve durable job state, stage prerequisites, and resume behavior before
  relying on unattended multi-worker deployments.

---

## API Endpoints

| Method | Endpoint | Status |
|---|---|---|
| `GET` | `/api/v1/health` | ✅ Working |
| `POST` | `/api/v1/research/query`, `/collect` | ✅ Query and collection |
| `POST` | `/api/v1/research/retrieve/{dense,sparse,hybrid,multi}` | ✅ Retrieval with required reranking |
| `POST` | `/api/v1/research/gaps` | ✅ Signal-based, non-LLM candidate detection |
| `POST` | `/api/v1/research/validate` | ✅ Evidence and coverage validation |
| `POST` | `/api/v1/research/kg/build` | ✅ Knowledge-graph build |
| `GET` | `/api/v1/research/*/status/{job_id}` | ✅ Background job status |

See [docs/gaps_kg_api.md](docs/gaps_kg_api.md) and
[docs/retrieval_api.md](docs/retrieval_api.md) for request requirements,
configuration, and error behavior.

---

## Key Design Decisions

1. **Modular monolith** — extensible without premature microservices complexity.
2. **Dependency injection** — all components receive deps through constructors; nothing is hard-wired.
3. **Storage abstraction** — `VectorStore`, `PaperRepository`, `EvidenceRepository` interfaces allow plug-in backends.
4. **LLM abstraction** — `LLMProvider` and `EmbeddingProvider` interfaces; no direct SDK calls in domain code.
5. **EWRGV is a distinct module** — `validation/` is separate from `gap_detection/` to reflect the research boundary.
6. **No premature optimisation** — No Celery, Redis, Kubernetes, Neo4j, or GraphRAG at this stage.
