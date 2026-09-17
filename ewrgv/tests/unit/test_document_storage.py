"""
tests/unit/test_document_storage.py
--------------------------------------
Unit tests for app/storage/document_storage.py (Phase 2.8).

Tests verify:
- upload returns canonical path on success
- upload raises StorageError on failure (DB status must NOT be updated)
- download retrieves bytes
- exists() returns True/False without raising on missing file
- delete() is idempotent
- get_path() returns deterministic path without I/O
- Security: no credentials logged
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
import httpx

from app.storage.artifact_paths import ArtifactType, get_artifact_path
from app.storage.document_storage import SupabaseDocumentStorage
from app.core.exceptions import StorageError


PAPER_ID = "test-paper-id-abc123"
BUCKET = "papers"


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


@pytest.fixture
def mock_storage_client():
    """A SupabaseStorageClient with mocked HTTP methods."""
    with patch("app.storage.document_storage.SupabaseStorageClient") as MockClient:
        client_instance = MagicMock()
        MockClient.return_value = client_instance

        # Mock upload and download
        client_instance.upload = AsyncMock(return_value="papers/test-paper-id-abc123/parsed.json")
        client_instance.download = AsyncMock(return_value=b'{"key": "value"}')

        # Mock the internal _client for exists/delete (HEAD/DELETE requests)
        inner_http = AsyncMock()
        client_instance._client = inner_http
        client_instance._url = "https://test.supabase.co"
        client_instance._auth_headers = MagicMock(return_value={
            "Authorization": "Bearer REDACTED",
            "apikey": "REDACTED",
        })

        yield client_instance, MockClient


@pytest.fixture
def storage(mock_storage_client):
    """A SupabaseDocumentStorage instance with mocked internals."""
    with patch("app.storage.document_storage.SupabaseStorageClient") as MockClient:
        client_instance = mock_storage_client[0]
        MockClient.return_value = client_instance

        inner_http = AsyncMock()
        client_instance._client = inner_http
        client_instance._url = "https://test.supabase.co"
        client_instance._auth_headers = MagicMock(return_value={
            "Authorization": "Bearer REDACTED",
        })

        ds = SupabaseDocumentStorage(bucket=BUCKET)
        ds._client = client_instance
        yield ds


# ------------------------------------------------------------------ #
# get_path — no I/O
# ------------------------------------------------------------------ #


class TestGetPath:
    def test_get_path_returns_canonical_path(self, storage):
        path = storage.get_path(PAPER_ID, ArtifactType.PARSED_DOCUMENT)
        assert path == f"papers/{PAPER_ID}/parsed.json"

    def test_get_path_fulltext(self, storage):
        path = storage.get_path(PAPER_ID, ArtifactType.FULL_TEXT)
        assert path == f"papers/{PAPER_ID}/fulltext.pdf"

    def test_get_path_no_io(self, storage):
        """get_path must not make any network calls."""
        storage._client.upload = AsyncMock(side_effect=Exception("should not be called"))
        storage._client.download = AsyncMock(side_effect=Exception("should not be called"))
        # Should not raise
        path = storage.get_path(PAPER_ID, ArtifactType.CHUNKS)
        assert "chunks.json" in path


# ------------------------------------------------------------------ #
# upload
# ------------------------------------------------------------------ #


class TestUpload:
    @pytest.mark.asyncio
    async def test_upload_returns_path_on_success(self, storage):
        storage._client.upload = AsyncMock(return_value=None)
        path = await storage.upload(
            PAPER_ID, ArtifactType.PARSED_DOCUMENT, b"content", "application/json"
        )
        assert path == f"papers/{PAPER_ID}/parsed.json"

    @pytest.mark.asyncio
    async def test_upload_calls_client_with_correct_args(self, storage):
        storage._client.upload = AsyncMock(return_value=None)
        await storage.upload(
            PAPER_ID, ArtifactType.SECTIONS, b"data", "application/json"
        )
        storage._client.upload.assert_awaited_once_with(
            bucket=BUCKET,
            path=f"papers/{PAPER_ID}/sections.json",
            content=b"data",
            content_type="application/json",
            upsert=True,
        )

    @pytest.mark.asyncio
    async def test_upload_raises_storage_error_on_failure(self, storage):
        """
        Critical: upload raises StorageError on failure.
        The caller must NOT update DB status unless this succeeds.
        """
        storage._client.upload = AsyncMock(
            side_effect=StorageError("Upload failed", detail="HTTP 500")
        )
        with pytest.raises(StorageError):
            await storage.upload(
                PAPER_ID, ArtifactType.PARSED_DOCUMENT, b"data", "application/json"
            )

    @pytest.mark.asyncio
    async def test_upload_idempotent_upsert_true(self, storage):
        """Second upload overwrites cleanly (upsert=True)."""
        storage._client.upload = AsyncMock(return_value=None)
        path1 = await storage.upload(PAPER_ID, ArtifactType.CHUNKS, b"v1", "application/json")
        path2 = await storage.upload(PAPER_ID, ArtifactType.CHUNKS, b"v2", "application/json")
        assert path1 == path2  # Same canonical path both times


# ------------------------------------------------------------------ #
# download
# ------------------------------------------------------------------ #


class TestDownload:
    @pytest.mark.asyncio
    async def test_download_returns_bytes(self, storage):
        expected = b'{"test": true}'
        storage._client.download = AsyncMock(return_value=expected)
        result = await storage.download(PAPER_ID, ArtifactType.PARSED_DOCUMENT)
        assert result == expected

    @pytest.mark.asyncio
    async def test_download_calls_client_with_correct_path(self, storage):
        storage._client.download = AsyncMock(return_value=b"data")
        await storage.download(PAPER_ID, ArtifactType.CHUNKS)
        storage._client.download.assert_awaited_once_with(
            bucket=BUCKET,
            path=f"papers/{PAPER_ID}/chunks.json",
        )

    @pytest.mark.asyncio
    async def test_download_raises_storage_error_when_missing(self, storage):
        storage._client.download = AsyncMock(
            side_effect=StorageError("Not found", detail="HTTP 404")
        )
        with pytest.raises(StorageError):
            await storage.download(PAPER_ID, ArtifactType.FULL_TEXT)


# ------------------------------------------------------------------ #
# exists
# ------------------------------------------------------------------ #


class TestExists:
    @pytest.mark.asyncio
    async def test_exists_returns_true_on_200(self, storage):
        mock_response = MagicMock()
        mock_response.status_code = 200
        storage._client._client.head = AsyncMock(return_value=mock_response)
        result = await storage.exists(PAPER_ID, ArtifactType.PARSED_DOCUMENT)
        assert result is True

    @pytest.mark.asyncio
    async def test_exists_returns_false_on_404(self, storage):
        mock_response = MagicMock()
        mock_response.status_code = 404
        storage._client._client.head = AsyncMock(return_value=mock_response)
        result = await storage.exists(PAPER_ID, ArtifactType.PARSED_DOCUMENT)
        assert result is False

    @pytest.mark.asyncio
    async def test_exists_returns_false_on_400(self, storage):
        """400 is also treated as 'not found' (Supabase returns 400 for some missing objects)."""
        mock_response = MagicMock()
        mock_response.status_code = 400
        storage._client._client.head = AsyncMock(return_value=mock_response)
        result = await storage.exists(PAPER_ID, ArtifactType.CHUNKS)
        assert result is False

    @pytest.mark.asyncio
    async def test_exists_raises_storage_error_on_unexpected_status(self, storage):
        mock_response = MagicMock()
        mock_response.status_code = 500
        storage._client._client.head = AsyncMock(return_value=mock_response)
        with pytest.raises(StorageError):
            await storage.exists(PAPER_ID, ArtifactType.SECTIONS)

    @pytest.mark.asyncio
    async def test_exists_raises_storage_error_on_network_error(self, storage):
        storage._client._client.head = AsyncMock(
            side_effect=httpx.RequestError("Connection refused")
        )
        with pytest.raises(StorageError):
            await storage.exists(PAPER_ID, ArtifactType.FULL_TEXT)


# ------------------------------------------------------------------ #
# delete
# ------------------------------------------------------------------ #


class TestDelete:
    @pytest.mark.asyncio
    async def test_delete_success_on_200(self, storage):
        mock_response = MagicMock()
        mock_response.status_code = 200
        storage._client._client.delete = AsyncMock(return_value=mock_response)
        # Should not raise
        await storage.delete(PAPER_ID, ArtifactType.CHUNKS)

    @pytest.mark.asyncio
    async def test_delete_idempotent_on_404(self, storage):
        """Deleting a non-existent artifact should not raise."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        storage._client._client.delete = AsyncMock(return_value=mock_response)
        # Should not raise
        await storage.delete(PAPER_ID, ArtifactType.PARSED_DOCUMENT)

    @pytest.mark.asyncio
    async def test_delete_raises_storage_error_on_unexpected_status(self, storage):
        mock_response = MagicMock()
        mock_response.status_code = 500
        storage._client._client.delete = AsyncMock(return_value=mock_response)
        with pytest.raises(StorageError):
            await storage.delete(PAPER_ID, ArtifactType.FULL_TEXT)


# ------------------------------------------------------------------ #
# Status-consistency guarantee
# ------------------------------------------------------------------ #


class TestStatusConsistency:
    """
    Verify the caller pattern: upload first, update DB only on success.

    The storage abstraction does NOT update the DB — that is always the
    caller's responsibility.  These tests verify that a StorageError is
    raised BEFORE the caller would normally update the DB.
    """

    @pytest.mark.asyncio
    async def test_storage_failure_does_not_silently_succeed(self, storage):
        """
        If upload fails, caller receives StorageError.
        The caller MUST NOT update DB status to PARSED/CHUNKED/etc.
        """
        storage._client.upload = AsyncMock(
            side_effect=StorageError("Bucket unavailable")
        )
        db_status_updated = False

        try:
            await storage.upload(
                PAPER_ID, ArtifactType.PARSED_DOCUMENT, b"data", "application/json"
            )
            # This line should NEVER be reached
            db_status_updated = True
        except StorageError:
            pass  # Expected

        assert db_status_updated is False, (
            "DB status must not be updated if storage upload fails"
        )


# ------------------------------------------------------------------ #
# Security: no credentials logged
# ------------------------------------------------------------------ #


class TestSecurity:
    def test_get_path_does_not_expose_key(self, storage, caplog):
        """get_path must not log any auth headers or keys."""
        import logging
        with caplog.at_level(logging.DEBUG):
            storage.get_path(PAPER_ID, ArtifactType.FULL_TEXT)
        for record in caplog.records:
            assert "Bearer" not in record.getMessage()
            assert "REDACTED" not in record.getMessage() or "Bearer REDACTED" not in record.getMessage()
