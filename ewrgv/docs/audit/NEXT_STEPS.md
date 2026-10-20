# Prioritized next steps

**Completed follow-up (2026-10-20):** Public retrieval/validation and `/gaps`
now require CrossEncoder reranking; `/gaps` uses bounded reranked query context
with non-LLM signal detectors and rejects LLM options. README, retrieval/gap
API docs and audit records were updated. Continue with evaluation and runtime
reliability work below.

This sequence addresses correctness before expanding methodology. It records remaining work after the completed retrieval/gap API follow-up above.

## Priority 0 — certify persistence and results integrity

**Dependency:** disposable PostgreSQL/Supabase-compatible test database and current migrations. **Files:** `storage/repositories/kg_repository.py`, `scripts/migrations/`, `validation/jobs.py`, `knowledge_graph/jobs.py`.

- Reproduce the historic case where a candidate row exists but validation fields remain null.
- Make validation completion contingent on committed, non-empty or explicitly empty result persistence; detect zero-row updates and expose a failed status with a useful error.
- Decide whether validation reruns clear old results before success or preserve the prior snapshot until replacement commits.

**Acceptance:** tests cover missing result row, zero-row update, transaction rollback, successful persist/reload, repeated validation and job status. `GET /research/validate/{run_id}` returns a contractually defined response for absent, running, failed and completed results.

## Priority 1 — schema/config compatibility gate

**Dependency:** choose the supported database. **Files:** `app/core/config.py`, `storage/relational/database.py`, migrations 01–11, embedding factory/repository.

- Resolve the SQLite-looking default versus asyncpg/PostgreSQL-only migrations.
- Align embedding provider/model/dimension with migration 08's `vector(384)` and invalidate/re-embed on model change.
- Add startup checks for required DSN, Supabase object storage, extension, schema version and expected vector dimension.

**Acceptance:** fresh disposable Postgres database migrates from zero; all repositories run against it; incompatible config fails at startup with a redacted actionable error; embedding dimension mismatch is caught before processing.

## Priority 2 — durable stage/job workflow

**Dependency:** stable DB and job result contract. **Files:** `core/job_store.py`, routes, processing workers and UI polling code.

- Persist job/stage states and aggregate run state; implement prerequisites and retry/resume behavior.
- Choose supported deployment scale (single process or multiworker). For multiworker/restart survival, use a durable queue and shared status store.
- Define idempotency keys and what each rerun replaces or preserves.

**Acceptance:** restart during each stage leaves a recoverable state; polling from another worker returns same status; a duplicate request does not corrupt artifacts or create unintended duplicate runs; transitions and terminal failure are documented.

## Priority 3 — freeze API contracts for UI

**Dependency:** priorities 0–2 or an explicit MVP scope. **Files:** route schemas and `docs/retrieval_api.md`, `docs/gaps_kg_api.md`, UI README/client.

- Generate/review OpenAPI for all routes; standardize job response shape, errors, pagination and result-absent semantics.
- Decide if current uncommitted paper sorting is part of the supported contract.
- Verify completed-run restoration and all UI empty/error/partial-success paths.

**Acceptance:** contract tests enforce examples and status codes; UI supports existing completed, partially processed, failed and new runs after browser/server reload; no required result request silently becomes a new run.

## Priority 4 — pipeline quality/evaluation

**Dependency:** reproducible stable data flow. **Files:** `gap_detection/signals/*`, `gap_detection/evidence_scoring.py`, `validation/*`, benchmark docs.

- Create fixed, versioned literature/query test corpus and expert-labeled candidate gaps.
- Evaluate retrieval recall, candidate precision/recall, evidence grounding, false-positive rate and validation calibration against baselines.
- Report effects of incomplete OA text, corpus size, citation age, terminology and provider failures.

**Acceptance:** repeatable benchmark with seeded/config/versioned inputs, metrics and confidence intervals; no novelty or accuracy claim without results.

## Priority 5 — reliability and security hardening

**Dependency:** intended hosting model. **Files:** `main.py`, config, HTTP clients, worker/repository layers.

- Add auth/authorization or document trusted reverse-proxy boundary; constrain CORS for deployment.
- Review retries, backoff, provider rate limits, cancellation/timeouts, resource caps and secret redaction.
- Add structured job/run correlation logs and operational metrics; close pools/clients on shutdown.

**Acceptance:** secrets absent from errors/logs; provider outage cannot yield false completion; logs identify run/job/stage without paper content leakage; resource limits are tested on large runs.

## Recommended next three development tasks

1. **Reproduce and fix validation result persistence** in a disposable Postgres integration test, including zero-row update and run reload behavior.
2. **Resolve and enforce database/embedding configuration** (explicit Postgres DSN and 384-dimensional model/migration agreement).
3. **Define durable run/job state and prerequisites** so the UI can resume, poll and display existing results reliably.

## Backend readiness for UI integration

The UI can integrate against the current endpoints for a controlled single-process demonstration. For a reliable user workflow, stabilize these contracts first: (1) aggregate run/stage state and durable job status; (2) validation persistence and absent-result semantics; (3) stable request/response schemas for query, collection and job starts/status; (4) partial provider failure and retry semantics; (5) compatibility of paper list sorting/pagination. Current UI code is untracked and its API sorting change is uncommitted, so those capabilities are not yet part of the repository baseline.
