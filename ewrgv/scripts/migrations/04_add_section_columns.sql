-- Migration 04: Add Section Extraction Columns (Phase 2.6)
--
-- section_status    tracks the section-extraction pipeline stage.
-- parsing_status    (Phase 2.5) and full_text_status (Phase 2.4) are distinct.
-- These columns must NOT be conflated.

ALTER TABLE papers
ADD COLUMN IF NOT EXISTS section_status          TEXT  DEFAULT 'NOT_EXTRACTED',
ADD COLUMN IF NOT EXISTS sections_storage_path   TEXT,
ADD COLUMN IF NOT EXISTS sections_count          INTEGER,
ADD COLUMN IF NOT EXISTS sectioned_at            TIMESTAMPTZ;

-- Efficient lookup for papers pending or eligible for section extraction
CREATE INDEX IF NOT EXISTS idx_papers_section_status ON papers(section_status);
