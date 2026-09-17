-- Migration 07: Embedding Table with pgvector (Phase 2.9)
--
-- Adds:
-- 1. pgvector extension for vector storage
-- 2. chunk_embeddings table for persisting dense embeddings
-- 3. embedding_status column on papers table
--
-- Requirements:
--   Supabase PostgreSQL must support the pgvector extension.
--   This is available on all Supabase projects (free tier included).

-- ------------------------------------------------------------------ --
-- 1. Enable pgvector extension
-- ------------------------------------------------------------------ --
CREATE EXTENSION IF NOT EXISTS vector;

-- ------------------------------------------------------------------ --
-- 2. Chunk embeddings table
-- ------------------------------------------------------------------ --
--
-- Design notes:
-- * PRIMARY KEY (chunk_id, embedding_model) prevents duplicate embeddings
--   for the same chunk under the same model.
-- * embedding column uses vector(1536) matching text-embedding-3-small.
--   If a different model is used, the column can store any dimension ≤ 2000.
-- * paper_id is denormalized for efficient per-paper queries.
-- * No IVFFlat/HNSW index is created here — that belongs to Phase 2.10
--   (retrieval) when the query pattern is known.
--
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id            TEXT NOT NULL,
    paper_id            TEXT NOT NULL,
    embedding           vector(1536) NOT NULL,
    embedding_model     TEXT NOT NULL,
    embedding_dimension INTEGER NOT NULL,
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    updated_at          TIMESTAMPTZ DEFAULT NOW(),

    PRIMARY KEY (chunk_id, embedding_model)
);

-- Indexes for efficient lookups
CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_paper_id
    ON chunk_embeddings(paper_id);

CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_model
    ON chunk_embeddings(embedding_model);

-- ------------------------------------------------------------------ --
-- 3. Embedding status column on papers table
-- ------------------------------------------------------------------ --
ALTER TABLE papers
ADD COLUMN IF NOT EXISTS embedding_status TEXT DEFAULT 'NOT_EMBEDDED';

ALTER TABLE papers
ADD COLUMN IF NOT EXISTS embedded_at TIMESTAMPTZ;

ALTER TABLE papers
ADD COLUMN IF NOT EXISTS embedded_chunks_count INTEGER DEFAULT 0;
