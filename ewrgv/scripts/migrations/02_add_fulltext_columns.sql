-- Migration 02: Add Full-Text Acquisition Columns (Phase 2.4)

ALTER TABLE papers
ADD COLUMN IF NOT EXISTS full_text_status TEXT DEFAULT 'NOT_CHECKED',
ADD COLUMN IF NOT EXISTS full_text_url TEXT,
ADD COLUMN IF NOT EXISTS storage_path TEXT,
ADD COLUMN IF NOT EXISTS source_type TEXT,
ADD COLUMN IF NOT EXISTS acquired_at TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS content_type TEXT,
ADD COLUMN IF NOT EXISTS file_size INTEGER;

-- Create an index to quickly find papers pending acquisition
CREATE INDEX IF NOT EXISTS idx_papers_full_text_status ON papers(full_text_status);
