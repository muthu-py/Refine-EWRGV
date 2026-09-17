-- Migration 03: Add Document Parsing Columns (Phase 2.5)
--
-- Adds parsing state separately from full-text acquisition state.
-- full_text_status tracks acquisition (Phase 2.4).
-- parsing_status   tracks document parsing  (Phase 2.5).
-- These are distinct pipeline stages and must NOT be conflated.

ALTER TABLE papers
ADD COLUMN IF NOT EXISTS parsing_status       TEXT         DEFAULT 'NOT_PARSED',
ADD COLUMN IF NOT EXISTS parsed_storage_path  TEXT,
ADD COLUMN IF NOT EXISTS page_count           INTEGER,
ADD COLUMN IF NOT EXISTS parsed_at            TIMESTAMPTZ;

-- Index for efficient lookups of papers pending or eligible for parsing
CREATE INDEX IF NOT EXISTS idx_papers_parsing_status ON papers(parsing_status);
