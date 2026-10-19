-- Migration 11: persisted Phase 4 validation results and external-search cache.
-- Safe to re-run; cache is shared by validation and temporal trend searches.

ALTER TABLE gap_detection_results
    ADD COLUMN IF NOT EXISTS validation_params JSONB,
    ADD COLUMN IF NOT EXISTS validation_stats JSONB,
    ADD COLUMN IF NOT EXISTS validation_results JSONB,
    ADD COLUMN IF NOT EXISTS validated_at TIMESTAMPTZ;

CREATE TABLE IF NOT EXISTS external_search_cache (
    key         TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    response    JSONB NOT NULL,
    fetched_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_external_search_cache_fetched_at
    ON external_search_cache (fetched_at);
