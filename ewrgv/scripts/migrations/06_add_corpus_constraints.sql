-- Migration 06: Corpus Storage Constraints (Phase 2.8)
--
-- Adds:
-- 1. Unique constraint on paper_sources to prevent duplicate provenance rows.
--    A paper can have multiple sources, but the same (paper_id, run_id, provider,
--    provider_paper_id) combination should appear at most once.
--
-- 2. Composite index on processing status columns for efficient health queries.
--    Supports: COUNT(*) FILTER (WHERE full_text_status = ...) patterns.
--
-- 3. Check constraints to validate that processing status columns only contain
--    known values. This prevents typos from silently entering the DB.

-- ------------------------------------------------------------------ --
-- 1. Unique provenance constraint
-- ------------------------------------------------------------------ --
-- Prevents the same (paper, run, provider, provider_paper_id) combination
-- from being inserted more than once, while still allowing:
--   - Same paper in multiple runs (different run_id)
--   - Same paper from multiple providers in the same run
--   - Multiple queries to the same provider for the same paper
--     (different provider_paper_id is NOT expected; same provider_paper_id
--     would be the duplicate we're preventing)
--
-- NOTE: provider_paper_id may be NULL for papers without provider IDs.
-- The constraint uses NULLS NOT DISTINCT so two rows with NULL provider_paper_id
-- for the same (paper_id, run_id, provider) are treated as duplicates.
ALTER TABLE paper_sources
ADD CONSTRAINT IF NOT EXISTS uq_paper_sources_provenance
    UNIQUE NULLS NOT DISTINCT (paper_id, run_id, provider, provider_paper_id);

-- ------------------------------------------------------------------ --
-- 2. Composite index for corpus health queries
-- ------------------------------------------------------------------ --
-- Speeds up COUNT(*) FILTER (WHERE ...) status aggregation queries
-- used by PaperRepository.get_corpus_health_counts().
CREATE INDEX IF NOT EXISTS idx_papers_processing_statuses
    ON papers (full_text_status, parsing_status, section_status, chunk_status);

-- ------------------------------------------------------------------ --
-- 3. Processing status check constraints
-- ------------------------------------------------------------------ --

ALTER TABLE papers
ADD CONSTRAINT IF NOT EXISTS chk_full_text_status
    CHECK (full_text_status IN (
        'NOT_CHECKED', 'OA_FOUND', 'NO_OA_FOUND',
        'DOWNLOAD_FAILED', 'DOWNLOADED', 'INVALID_DOCUMENT'
    ));

ALTER TABLE papers
ADD CONSTRAINT IF NOT EXISTS chk_parsing_status
    CHECK (parsing_status IN (
        'NOT_PARSED', 'PARSED', 'PARSE_FAILED', 'INVALID_DOCUMENT'
    ));

ALTER TABLE papers
ADD CONSTRAINT IF NOT EXISTS chk_section_status
    CHECK (section_status IN (
        'NOT_EXTRACTED', 'EXTRACTED', 'EXTRACTION_FAILED'
    ));

ALTER TABLE papers
ADD CONSTRAINT IF NOT EXISTS chk_chunk_status
    CHECK (chunk_status IN (
        'NOT_CHUNKED', 'CHUNKED', 'CHUNKING_FAILED'
    ));
