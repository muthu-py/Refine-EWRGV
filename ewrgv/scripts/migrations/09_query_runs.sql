-- Migration 09: persist query understanding and link it to research runs
--
-- Additive only.
-- 1. query_runs stores the output of POST /research/query (question, query
--    understanding, expanded queries) so /collect can run every expanded
--    query by query_id and later stages can reuse the concept dictionary.
-- 2. research_runs.query_id links each collection run to its query.

CREATE TABLE IF NOT EXISTS query_runs (
    query_id          UUID PRIMARY KEY,
    research_question TEXT NOT NULL,
    query_type        TEXT,
    understanding     JSONB,
    expanded_queries  JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE research_runs
ADD COLUMN IF NOT EXISTS query_id UUID REFERENCES query_runs(query_id);

CREATE INDEX IF NOT EXISTS idx_research_runs_query_id ON research_runs(query_id);
