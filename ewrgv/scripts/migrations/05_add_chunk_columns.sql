-- Migration 05: Add Chunking Columns (Phase 2.7)
--
-- chunk_status      tracks the chunking pipeline stage.
-- sections_count    number of sections extracted (Phase 2.6, stored here for convenience).
-- chunks_storage_path, chunks_count, chunked_at are Phase 2.7 output.
--
-- Note: section_status, sections_storage_path, sections_count, sectioned_at
--       are already added by migration 04.

ALTER TABLE papers
ADD COLUMN IF NOT EXISTS chunk_status          TEXT   DEFAULT 'NOT_CHUNKED',
ADD COLUMN IF NOT EXISTS chunks_storage_path   TEXT,
ADD COLUMN IF NOT EXISTS chunks_count          INTEGER,
ADD COLUMN IF NOT EXISTS chunked_at            TIMESTAMPTZ;

-- Efficient lookup for papers pending or eligible for chunking
CREATE INDEX IF NOT EXISTS idx_papers_chunk_status ON papers(chunk_status);
