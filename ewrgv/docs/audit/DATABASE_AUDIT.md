# Database audit

## Scope and caveat

This audit read SQL migrations, database configuration and repository source only. It did **not** connect to PostgreSQL/Supabase, inspect a live schema, read credentials, or modify a database. Migration history is the schema evidence; deployed schema may differ.

## Persistence components

- **Relational store:** `app/storage/relational/database.py` uses `asyncpg` connection pooling. Settings expose `DATABASE_URL`, `SUPABASE_DATABASE_URL`, and Supabase credentials; do not infer which is active without inspecting runtime code/config. The default `DATABASE_URL` is `sqlite:///./data/artifacts/ewrgv.db`, but `asyncpg` cannot use that URL and the migrations rely on PostgreSQL/pgvector. Deployment needs an explicit PostgreSQL DSN.
- **Object storage:** `app/storage/supabase_storage.py`, bucket defaults to `papers`; acquisition and parsed/section/chunk artifacts are stored as objects. `document_storage.py` and `artifact_paths.py` handle local/artifact addressing as well.
- **Embeddings:** `chunk_embeddings` stores vectors in Postgres pgvector; dense retrieval reads via repository code. Migration 07 originally specified 1536 dimensions; migration 08 changes it to 384 for local `BAAI/bge-small-en-v1.5` and resets embedding state.
- **In-memory state:** `JobStore` is not database-backed. Corpus cache, provider caches and BM25 artifacts are separate from relational truth.

## Tables from migrations

| Table | Keys and important fields | Relationships / purpose |
|---|---|---|
| `research_runs` (01; extended 09) | `run_id UUID PK`, `research_question`, `created_at`, later nullable `query_id` | One collection/research run; linked to `query_runs` by query ID (not shown as enforced FK). |
| `papers` (01 + 02–08) | `paper_id UUID PK`; `title`, `abstract`, unique `doi`, `publication_year`, `authors JSONB`; acquisition fields (`full_text_status`, URL/path/size/time/error); parse fields (`parsing_status`, `parsed_storage_path`, page count/time); section fields; chunk fields; embedding status/time/count | Canonical deduplicated paper row referenced by `paper_sources`. Processing columns are per paper, not per run. |
| `paper_sources` (01; constraints in 06) | `source_id UUID PK`, `paper_id`, `run_id`, provider/provider ID, source query, URL, citations, created time | Provenance junction between a paper and a run/provider. FKs cascade from canonical paper/run. Migration 06 adds uniqueness/constraints and indexes; inspect exact SQL for deployed dedup semantics. |
| `query_runs` (09) | `query_id UUID PK`, question, type, understanding JSONB, expanded queries JSONB, created time | Stores query decomposition used by collection and later stages. `research_runs.query_id` is added/indexed. |
| `chunk_embeddings` (07 then 08) | Composite PK `(chunk_id, embedding_model)`, `paper_id`, `embedding vector(384)` after 08, dimension/model/timestamps | Chunk vectors; no declared FK to artifact/chunk tables in migrations reviewed. Indexes on paper/model, but no ANN index is created. |
| `kg_graphs` (10) | `run_id TEXT PK`, `query_id`, built time, `stats JSONB`, `payload JSONB` | Latest serialized graph per run; rebuild replaces prior run row. No FK shown. |
| `gap_detection_results` (10; extended 11) | `run_id TEXT PK`, `query_id`, created time, params/stats/candidates JSONB, optional baseline JSONB; 11 adds validation results/metadata and `validated_at` | Latest detection/validation payload per run; data is JSONB, one row per run. No FK shown. |
| `external_search_cache` (11) | `key TEXT PK`, source, response JSONB, fetched_at | Cached external search response; indexed by fetched time. |

`papers` state vocabulary comes from worker and migration code: full-text (`NOT_FOUND`/`DOWNLOADED`/failed variants), parsing (`NOT_PARSED`/`PARSED`/failure), section, chunk and embedding statuses. Exact allowed values are not enforced as database enums in the inspected migrations. Code and DB can drift silently unless checked by migrations/integration tests.

The migration files explicitly add `parsing_status` in migration 03, and the parsing worker uses parsing-stage state; this static audit did not establish a current source-to-migration column-name mismatch. A deployed schema mismatch remains unverified because the live database was not inspected. The more visible current inconsistency is embedding configuration: settings default to OpenAI `text-embedding-3-small` at 1536 dimensions, while migration 08 changes the stored vector column to 384 for local BGE embeddings. The factory may be configured differently in deployment, but the repository defaults and migration are not aligned.

## Migration sequence and schema drift risks

1. `01_create_papers_table.sql`: research run, canonical papers, provenance junction.
2. `02_add_fulltext_columns.sql`; `03_add_parsing_columns.sql`; `04_add_section_columns.sql`; `05_add_chunk_columns.sql`: progressive processing states and indexes.
3. `06_add_corpus_constraints.sql`: source uniqueness and processing constraints/indexes.
4. `07_add_embedding_table.sql`: pgvector extension, 1536 vector schema.
5. `08_embeddings_384.sql`: destructive-to-data migration by design: removes incompatible prior embeddings and resets paper embedding status; dimension 384.
6. `09_query_runs.sql`: query history and link.
7. `10_kg_and_gaps.sql`: serialized run graph/candidate result tables.
8. `11_gap_validation.sql`: validation columns and shared external search cache.

The migration runner lives in `scripts/{run_migrations,apply_migrations}.py`. Inspect its ledger/ordering and operators before deployments; the audit did not execute it. Versioned schema has no ORM model layer: persistence mappings and SQL are in repositories. This increases the need to keep migrations, SQL strings, status fields and test fixtures synchronized.

## Relationships and transaction concerns

`research_runs` → `paper_sources` → `papers` is the primary relational path. A paper can appear in multiple runs/providers. Query run links to research run; graph and gap records use `run_id` keys and JSONB payloads. Chunk/artifact associations are primarily file/object references plus IDs rather than normalized tables. Embeddings have `(chunk_id, model)` uniqueness.

Transactions and idempotency should be verified at repository write boundaries. Collection may touch run, paper, and source records; processing workers update paper-stage fields and object paths; detection and validation replace JSONB values per run. Audit did not establish whether each multi-write path has an explicit transaction. One unchecked update is confirmed: `KnowledgeGraphRepository.save_validations()` ignores the command tag returned from its `UPDATE gap_detection_results ... WHERE run_id=$1`. The validation worker can therefore mark a job completed even when no row matched. This is consistent with the historic report, but no live row was inspected.

## Historic status issue from conversation

The 2026-10-19 project conversation included a user report that two rows had candidate detections but `validation_results` and `validated_at` were null, causing `GET /research/validate/{run_id}` to return 404. Current repository source confirms the matching design defect: validation UPDATE does not check whether it matched a row. This is not a live database observation. Reproduce against a disposable test database, inspect affected-row behavior and correlate with analysis job state.

## Required schema acceptance checks

- Fresh database executes all migrations in order; applying twice is safe where claimed.
- Current embedding provider dimension equals pgvector column dimension and `embedding_dimension` values.
- All repository SELECT/INSERT/UPDATE column names match migrations, especially parsing/section/chunk/embedding and validation fields.
- Each job completion implies persisted output exists; zero-row updates become failure.
- FK/cascade, unique keys and JSON payload versioning are defined and tested.
- Production config explicitly uses PostgreSQL; reject incompatible SQLite URL at startup instead of failing during a request.
