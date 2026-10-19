-- Migration 08: switch chunk_embeddings to 384-dim vectors
--
-- For the local sentence-transformers model BAAI/bge-small-en-v1.5 (384 dims),
-- replacing text-embedding-3-small / gemini-embedding-001 (1536 dims).
--
-- Existing vectors cannot be cast to a different dimension, so they are
-- deleted and every paper is reset to NOT_EMBEDDED; re-run /represent to
-- regenerate them.
--
-- Idempotent: does nothing once the column is already vector(384), so
-- re-running scripts/apply_migrations.py never wipes new embeddings.

DO $$
BEGIN
    IF (
        SELECT format_type(a.atttypid, a.atttypmod)
        FROM pg_attribute a
        WHERE a.attrelid = 'public.chunk_embeddings'::regclass
          AND a.attname = 'embedding'
          AND NOT a.attisdropped
    ) IS DISTINCT FROM 'vector(384)' THEN
        DELETE FROM chunk_embeddings;
        ALTER TABLE chunk_embeddings ALTER COLUMN embedding TYPE vector(384);
        UPDATE papers SET embedding_status = 'NOT_EMBEDDED', embedded_at = NULL;
        RAISE NOTICE 'chunk_embeddings.embedding migrated to vector(384)';
    END IF;
END
$$;
