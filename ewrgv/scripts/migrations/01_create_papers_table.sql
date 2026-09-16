-- Migration 01: Create EWRGV Collection Schema (Phase 2.2)

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. Research Runs
CREATE TABLE IF NOT EXISTS research_runs (
    run_id UUID PRIMARY KEY,
    research_question TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 2. Papers (Canonical)
CREATE TABLE IF NOT EXISTS papers (
    paper_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT NOT NULL,
    abstract TEXT,
    doi TEXT UNIQUE,
    publication_year INTEGER,
    authors JSONB DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 3. Paper Sources (Provenance Junction)
CREATE TABLE IF NOT EXISTS paper_sources (
    source_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    paper_id UUID NOT NULL REFERENCES papers(paper_id) ON DELETE CASCADE,
    run_id UUID NOT NULL REFERENCES research_runs(run_id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    provider_paper_id TEXT,
    source_query TEXT NOT NULL,
    paper_url TEXT,
    citation_count INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for efficient lookups
CREATE INDEX IF NOT EXISTS idx_papers_doi ON papers(doi);
CREATE INDEX IF NOT EXISTS idx_paper_sources_paper_id ON paper_sources(paper_id);
CREATE INDEX IF NOT EXISTS idx_paper_sources_run_id ON paper_sources(run_id);
CREATE INDEX IF NOT EXISTS idx_paper_sources_provider ON paper_sources(provider);
CREATE INDEX IF NOT EXISTS idx_paper_sources_provider_paper_id ON paper_sources(provider_paper_id);
