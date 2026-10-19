-- Migration 10: knowledge graph and candidate-gap results per run
--
-- Additive only.  One row per run in each table (latest build / detection
-- replaces the previous one).  Payloads are JSONB so the API can return
-- them without re-computation; see docs/gaps_kg_api.md.

CREATE TABLE IF NOT EXISTS kg_graphs (
    run_id      TEXT PRIMARY KEY,
    query_id    TEXT,
    built_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    stats       JSONB NOT NULL DEFAULT '{}'::jsonb,
    payload     JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS gap_detection_results (
    run_id      TEXT PRIMARY KEY,
    query_id    TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    params      JSONB NOT NULL DEFAULT '{}'::jsonb,
    stats       JSONB NOT NULL DEFAULT '{}'::jsonb,
    candidates  JSONB NOT NULL DEFAULT '[]'::jsonb,
    baseline    JSONB
);
