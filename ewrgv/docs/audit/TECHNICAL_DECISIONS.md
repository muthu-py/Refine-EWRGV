# Recovered technical decisions

Decision status here distinguishes code evidence from documentation/history. Alternatives are stated as options, not as decisions already made.

| Decision | Evidence and rationale | Alternatives / current question |
|---|---|---|
| Modular monolith with FastAPI | `app/` package splits API, domain, ingestion, retrieval, detection, validation and infrastructure. README/architecture docs name modular monolith. | Preserve for MVP. Split services only if independent scaling/deployment is needed. |
| Separate detection from validation | Distinct `gap_detection/` and `validation/` modules, `CandidateGap` vs `ValidationResult`, persisted detection and validation fields. This prevents a candidate-generation signal from being treated as confirmed absence. | Preserve as core research distinction. Clarify what validation source coverage is enough for each classification. |
| KG plus interpretable multi-signal candidates | `KnowledgeGraphBuilder` creates canonicalized entities/co-occurrence/claims; `GapDetectionService` fuses explicit, combination, temporal and topic signals. History identifies a preference for transparent rule/signal scoring before learned models. | ResearchRAG-style learned GAT would need labels and comparative evaluation. It is not implemented. |
| Hybrid retrieval with required API reranking | Retrieval service has BM25 and embedding-based semantic paths, fusion, and a local CrossEncoder reranker with candidate caps. API retrieval, validation and `/gaps` require reranking and fail closed. | Evaluate the reranker on a relevance benchmark, then tune caps/model and decide whether vector DB ANN indexing is necessary. Current retrieval and vector persistence are distinct concerns. |
| Provider abstraction | Search, LLM and embeddings have provider packages/factories. Configuration selects OpenRouter/OpenAI and OpenAI/local SentenceTransformers. | Confirm supported-provider names and credentials in docs; actual deployment choices are not recoverable without runtime config. |
| Supabase/Postgres persistence and Supabase object storage | Migrations use Postgres JSONB, UUID, TIMESTAMPTZ, pgvector; repositories use asyncpg; storage client handles documents. | Default SQLite DSN conflicts with these requirements. Choose one explicit deployment story; migration set implies Postgres. |
| SQL migrations rather than ORM schema | `scripts/migrations/*.sql` and direct repositories own schema behavior. | Keep if migrations become authoritative/versioned and verified. An ORM is not required but column drift tests are. |
| In-process BackgroundTasks/job status for early MVP | FastAPI route scheduling and `JobStore` avoids Celery/Redis operational setup. This matches old README statement that distributed queues were deferred. | Acceptable for one trusted process and short jobs. Before multi-worker/restart resilience, move to durable queue and DB-backed job state. |
| JSONB snapshots for KG, candidates, validation | Migrations 10–11 store latest run payload as JSONB, enabling simple retrieval and replacement. | Consider normalized evidence rows/versioned snapshots if querying, comparing or auditing result history becomes important. Current policy overwrites by run. |
| Signal-first public gap API with opt-in grounded LLM proposals and wording | `POST /gaps` defaults to signal detection; optional `include_llm_candidates` adds cited LLM proposals and optional `llm_wording` refines eligible signal wording while preserving originals. Sources stay explicit, LLM confidence is not scored, and all candidates still require validation. | Benchmark signal and LLM-assisted modes separately; audit cited-chunk provenance, deduplication, wording preservation, API-key errors and validation coverage. |
| Local BGE embeddings and 384 dimension | Migration 08 and comments name `BAAI/bge-small-en-v1.5`; config defaults elsewhere must be checked for effective values. | Align `EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_DIMENSION`, stored model version and DB vector type. Migration 07's 1536 notes are obsolete after 08. |
| No trained GNN or HySemRAG LDA pipeline | Source uses graph signals/topic clustering rather than trained GAT; history comparison cautioned against claiming those approaches. | Any method novelty claim requires literature comparison and evaluation, not architecture resemblance alone. |

## Conversation-derived product/engineering context

Accessible 2026-10-19 Codex history records the user describing the backend MVP as complete and requesting UI design. The user then reported issues: selecting a completed run opened a fresh pipeline; some saved DB rows had candidates but no validation; Semantic Scholar rate limiting appeared in an earlier UI discussion. Later code state contains UI changes and API changes, but this audit cannot assert every historic issue is resolved. Current `KnowledgeGraphRepository.save_validations()` confirms that its UPDATE does not inspect affected rows, consistent with the missing-validation report. No live database state was inspected.

An accessible methodology discussion describes `ResearchPipeline`'s LLM-based, injectable path alongside the KG signal-based API path. Current `/gaps` request validation fixes LLM options to false, and API retrieval is required to use the CrossEncoder. Earlier history about optional LLM API comparisons is superseded by this contract.

## Open architectural questions

1. Which database URL key is authoritative and what startup validation rejects SQLite/incompatible settings?
2. What corpus and expert-labelled benchmark will establish whether the signal detector and reranker produce useful candidates?
3. Should run-level job state, stage prerequisite state, and restart recovery be durable before wider UI use?
4. Are latest-per-run JSONB results sufficient, or should results be versioned to preserve reruns and explain changed findings?
5. What corpus adequacy and external source coverage are minimum conditions for a publishable `VALID_GAP` claim?
6. Which API contracts are supported/stable for UI, including pagination, sort, job stage errors, partial provider failures, and missing result semantics?
7. What evaluation corpus, expert labels and baselines will support claims about accuracy and research contribution?
