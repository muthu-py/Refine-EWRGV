"""
tests/unit/test_artifact_paths.py
-----------------------------------
Unit tests for app/storage/artifact_paths.py (Phase 2.8).

Tests verify:
- Canonical path format for all four artifact types
- NO double bucket prefix
- Determinism: same paper_id → same path
- Reverse lookup (parse_paper_id_from_path)
- Edge cases (empty paper_id, legacy paths)
"""

import pytest
from app.storage.artifact_paths import (
    ArtifactType,
    ARTIFACT_BUCKET,
    get_artifact_path,
    get_fulltext_path,
    get_parsed_path,
    get_sections_path,
    get_chunks_path,
    parse_paper_id_from_path,
)


SAMPLE_ID = "abc123-def456"
BUCKET = ARTIFACT_BUCKET  # "papers" by default


# ------------------------------------------------------------------ #
# Canonical path format
# ------------------------------------------------------------------ #


class TestCanonicalPaths:
    """All artifact paths must follow papers/{paper_id}/{filename}."""

    def test_fulltext_path(self):
        path = get_fulltext_path(SAMPLE_ID)
        assert path == f"papers/{SAMPLE_ID}/fulltext.pdf"

    def test_parsed_path(self):
        path = get_parsed_path(SAMPLE_ID)
        assert path == f"papers/{SAMPLE_ID}/parsed.json"

    def test_sections_path(self):
        path = get_sections_path(SAMPLE_ID)
        assert path == f"papers/{SAMPLE_ID}/sections.json"

    def test_chunks_path(self):
        path = get_chunks_path(SAMPLE_ID)
        assert path == f"papers/{SAMPLE_ID}/chunks.json"

    def test_get_artifact_path_all_types(self):
        """Verify all four ArtifactType values produce distinct, correct paths."""
        expected = {
            ArtifactType.FULL_TEXT: f"papers/{SAMPLE_ID}/fulltext.pdf",
            ArtifactType.PARSED_DOCUMENT: f"papers/{SAMPLE_ID}/parsed.json",
            ArtifactType.SECTIONS: f"papers/{SAMPLE_ID}/sections.json",
            ArtifactType.CHUNKS: f"papers/{SAMPLE_ID}/chunks.json",
        }
        for artifact_type, expected_path in expected.items():
            assert get_artifact_path(SAMPLE_ID, artifact_type) == expected_path


# ------------------------------------------------------------------ #
# No double bucket prefix
# ------------------------------------------------------------------ #


class TestNoBucketPrefix:
    """
    Verify the paths do NOT start with the bucket name as a prefix.

    The bucket name is supplied separately to the storage client.
    Supabase REST: POST /storage/v1/object/{bucket}/{path}

    Correct:  bucket="papers", path="papers/{id}/fulltext.pdf"
    Wrong:    bucket="papers", path="papers/papers/{id}/fulltext.pdf"  ← bug
    """

    def test_fulltext_does_not_start_with_double_bucket(self):
        path = get_fulltext_path(SAMPLE_ID)
        # Path starts with "papers/" (the folder), NOT "papers/papers/"
        assert path.startswith("papers/")
        assert not path.startswith(f"{BUCKET}/{BUCKET}/")

    def test_parsed_does_not_start_with_double_bucket(self):
        path = get_parsed_path(SAMPLE_ID)
        assert not path.startswith(f"{BUCKET}/{BUCKET}/")

    def test_sections_does_not_start_with_double_bucket(self):
        path = get_sections_path(SAMPLE_ID)
        assert not path.startswith(f"{BUCKET}/{BUCKET}/")

    def test_chunks_does_not_start_with_double_bucket(self):
        path = get_chunks_path(SAMPLE_ID)
        assert not path.startswith(f"{BUCKET}/{BUCKET}/")

    def test_path_structure_exactly_three_parts(self):
        """Path must be exactly 'papers/{paper_id}/{filename}' — three segments."""
        for artifact_type in ArtifactType:
            path = get_artifact_path(SAMPLE_ID, artifact_type)
            parts = path.split("/")
            assert len(parts) == 3, (
                f"Expected 3 path segments for {artifact_type}, got {parts}"
            )
            assert parts[0] == "papers"
            assert parts[1] == SAMPLE_ID


# ------------------------------------------------------------------ #
# Determinism
# ------------------------------------------------------------------ #


class TestDeterminism:
    """Same paper_id always produces the same path."""

    def test_same_paper_id_same_fulltext_path(self):
        assert get_fulltext_path(SAMPLE_ID) == get_fulltext_path(SAMPLE_ID)

    def test_different_paper_ids_different_paths(self):
        other_id = "zzzzzz-999999"
        assert get_fulltext_path(SAMPLE_ID) != get_fulltext_path(other_id)
        assert get_chunks_path(SAMPLE_ID) != get_chunks_path(other_id)

    def test_uuid_paper_id(self):
        """Verify UUID paper_ids work correctly."""
        uuid_id = "550e8400-e29b-41d4-a716-446655440000"
        path = get_fulltext_path(uuid_id)
        assert uuid_id in path
        assert path == f"papers/{uuid_id}/fulltext.pdf"


# ------------------------------------------------------------------ #
# Reverse lookup
# ------------------------------------------------------------------ #


class TestParseParperIdFromPath:

    def test_clean_path(self):
        path = f"papers/{SAMPLE_ID}/fulltext.pdf"
        assert parse_paper_id_from_path(path) == SAMPLE_ID

    def test_parsed_path(self):
        path = f"papers/{SAMPLE_ID}/parsed.json"
        assert parse_paper_id_from_path(path) == SAMPLE_ID

    def test_legacy_double_prefix_path(self):
        """Legacy bug: storage_path was stored as 'papers/papers/{id}/fulltext.pdf'."""
        legacy_path = f"papers/papers/{SAMPLE_ID}/fulltext.pdf"
        result = parse_paper_id_from_path(legacy_path)
        assert result == SAMPLE_ID

    def test_none_on_empty_path(self):
        assert parse_paper_id_from_path("") is None

    def test_none_on_wrong_structure(self):
        assert parse_paper_id_from_path("somethingwrong") is None

    def test_none_on_none_input(self):
        assert parse_paper_id_from_path(None) is None


# ------------------------------------------------------------------ #
# Error cases
# ------------------------------------------------------------------ #


class TestErrorCases:

    def test_empty_paper_id_raises(self):
        with pytest.raises(ValueError):
            get_artifact_path("", ArtifactType.FULL_TEXT)

    def test_all_artifact_filenames_are_unique(self):
        """No two artifact types should map to the same filename."""
        filenames = [
            get_artifact_path("x", t).split("/")[-1]
            for t in ArtifactType
        ]
        assert len(filenames) == len(set(filenames)), (
            f"Duplicate filenames detected: {filenames}"
        )
