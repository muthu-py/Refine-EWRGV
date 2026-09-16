# Query Understanding & Expansion (Phase 2.1)

## What it does

The **Query Understanding & Expansion** module takes a raw research question and uses an LLM to produce:

1. **A structured understanding** — domain, problem, task, methods, concepts, constraints, research intent, etc.
2. **Multiple expanded search queries** — diverse keyword-style formulations for high-recall literature discovery.

This is the first active stage of the EWRGV pipeline. It runs before any literature collection and its output drives the subsequent search strategy.

---

## Why it exists

A raw research question is rarely in the exact form needed to retrieve relevant papers. Different authors use different terminology, abbreviations, and framings for the same ideas. This module addresses that by:

- Decomposing the question into a machine-readable structured form.
- Generating complementary query formulations across terminology, method names, and domain perspectives.

The expanded queries are used in the next phase (Phase 2.2: Literature Collection) to query academic databases such as Semantic Scholar and OpenAlex.

---

## Input / Output Contract

### Input

A `ResearchQuery` with a non-empty `query` field:

```python
from app.domain.models.query import ResearchQuery
rq = ResearchQuery(query="How are GNNs used for CAN bus attack detection?")
```

### Output

An enriched `ResearchQuery` with:

- `understanding: QueryUnderstanding` — the structured decomposition
- `expanded_queries: list[str]` — diverse search query strings

The original `query` string is **always preserved unchanged**.

### QueryUnderstanding fields

| Field | Type | Description |
|---|---|---|
| `domain` | `str \| None` | High-level research domain |
| `problem` | `str \| None` | Central problem / challenge |
| `task` | `str \| None` | ML/research task |
| `methods` | `list[str]` | Methods/techniques mentioned or implied |
| `concepts` | `list[str]` | Key conceptual terms |
| `entities` | `list[str]` | Named entities (protocols, standards, products) |
| `datasets` | `list[str]` | Datasets explicitly mentioned |
| `metrics` | `list[str]` | Evaluation metrics explicitly mentioned |
| `constraints` | `list[str]` | Scope constraints (geography, population, etc.) |
| `research_intent` | `str \| None` | High-level intent classification |
| `terminology_variants` | `list[str]` | Alternative terms for broader search |
| `method_variants` | `list[str]` | Alternative method names |

---

## Architecture

```
ResearchPipeline / API route
         ↓
QueryUnderstandingService          ← app/query_understanding/service.py
         ↓
LLMProvider (protocol)             ← app/domain/interfaces.py
         ↓
OpenAILLMProvider                  ← app/providers/llm/openai_provider.py
         ↓
QueryUnderstandingLLMOutput        ← app/query_understanding/schemas.py (validated)
         ↓
QueryUnderstanding (domain model)  ← app/domain/models/query_understanding.py
         ↓
ResearchQuery.understanding        ← app/domain/models/query.py
```

The LLM provider is **replaceable** — any class satisfying `LLMProvider` works.

---

## Configuration

All configuration is via environment variables (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | LLM backend (currently only `openai` is implemented) |
| `LLM_MODEL` | `gpt-4o` | Model identifier |
| `LLM_API_KEY` | _(required)_ | API key — must be set in `.env` |
| `LLM_TEMPERATURE` | `0.0` | Sampling temperature |
| `LLM_MAX_TOKENS` | `4096` | Max tokens in completion |
| `LLM_TIMEOUT` | `30` | HTTP timeout in seconds |
| `QUERY_EXPANSION_MAX_QUERIES` | `8` | Maximum expanded queries to return |

### Minimal .env setup

```env
LLM_API_KEY=sk-your-openai-api-key-here
LLM_MODEL=gpt-4o
```

---

## Running via the API

Start the server:

```bash
cd ewrgv
uvicorn app.main:app --reload
```

Submit a research question:

```bash
curl -X POST http://localhost:8000/api/v1/research/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How are graph neural networks used for CAN bus attack detection?"}'
```

Response includes `understanding` and `expanded_queries`.

---

## Running the tests

```bash
cd ewrgv
python -m pytest tests/ -v
```

Phase 2.1-specific tests:

```bash
python -m pytest tests/unit/test_query_understanding_schemas.py \
                 tests/unit/test_query_understanding_service.py \
                 tests/unit/test_query_understanding_prompts.py \
                 -v
```

All tests run **fully offline** — no LLM API key is required for the test suite.

---

## Connection to Later Phases

```
Phase 2.1  →  ResearchQuery.understanding + expanded_queries
Phase 2.2  →  Literature Collection uses expanded_queries to query databases
Phase 2.3+ →  Retrieval, Gap Detection, EWRGV Validation
```

The `understanding` field is available throughout the pipeline for downstream use (e.g., gap detection may use `methods` and `concepts` as anchors).

---

## Error Handling

| Situation | Behaviour |
|---|---|
| LLM provider error / timeout | `ProviderError` raised |
| Malformed JSON from LLM | `ProviderError` raised |
| Missing optional fields | Safe defaults (`None` / `[]`) |
| No `LLM_API_KEY` configured | API returns stub response; tests still pass |
