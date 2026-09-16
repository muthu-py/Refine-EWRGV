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

> **Architecture Phase — v0.1.0**

This repository contains the **initial project architecture and development foundation**.

The pipeline skeleton is in place; core implementations are not yet started.

| Component | Status |
|---|---|
| Domain models | ✅ Complete |
| Interfaces / Protocols | ✅ Complete |
| Orchestration skeleton | ✅ Complete |
| EWRGV validation skeleton | ✅ Complete |
| API skeleton (`/health` working) | ✅ Complete |
| Config & logging | ✅ Complete |
| Gap detection implementation | 🔲 Not started |
| Retrieval implementation | 🔲 Not started |
| EWRGV confidence formula | 🔲 Not started |
| LLM / embedding providers | 🔲 Not started |
| Literature collectors | 🔲 Not started |
| Frontend | 🔲 Not started |

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

## Development Roadmap

### Phase 1 — Architecture Foundation (current)
- [x] Project structure
- [x] Domain models
- [x] Interfaces
- [x] Pipeline skeleton
- [x] API skeleton + health endpoint
- [x] Core tests

### Phase 2 — Retrieval Implementation
- [ ] Semantic Scholar literature collector
- [ ] PDF text extraction + chunking
- [ ] OpenAI embedding provider
- [ ] FAISS vector store adapter
- [ ] BM25 retriever
- [ ] Hybrid retrieval + RRF fusion
- [ ] Cross-encoder reranker

### Phase 3 — Gap Detection
- [ ] LLM-based gap detector
- [ ] Structured gap representation
- [ ] Candidate ranking

### Phase 4 — EWRGV Validation
- [ ] Verification query generation
- [ ] Direct, terminology, method, counter searches
- [ ] Evidence classification
- [ ] Coverage analysis
- [ ] EWRGV confidence formula design + implementation
- [ ] Classification
- [ ] Evidence-chain explanation

### Phase 5 — Evaluation & Frontend
- [ ] Evaluation dataset creation
- [ ] Benchmark against baseline systems
- [ ] React frontend (API-first — backend ready)

---

## API Endpoints

| Method | Endpoint | Status |
|---|---|---|
| `GET` | `/api/v1/health` | ✅ Working |
| `POST` | `/api/v1/research/query` | 🔲 Placeholder |
| `POST` | `/api/v1/research/gaps` | 🔲 Placeholder |
| `POST` | `/api/v1/research/validate` | 🔲 Placeholder |
| `GET` | `/api/v1/research/{job_id}` | 🔲 Placeholder |

---

## Key Design Decisions

1. **Modular monolith** — extensible without premature microservices complexity.
2. **Dependency injection** — all components receive deps through constructors; nothing is hard-wired.
3. **Storage abstraction** — `VectorStore`, `PaperRepository`, `EvidenceRepository` interfaces allow plug-in backends.
4. **LLM abstraction** — `LLMProvider` and `EmbeddingProvider` interfaces; no direct SDK calls in domain code.
5. **EWRGV is a distinct module** — `validation/` is separate from `gap_detection/` to reflect the research boundary.
6. **No premature optimisation** — No Celery, Redis, Kubernetes, Neo4j, or GraphRAG at this stage.
