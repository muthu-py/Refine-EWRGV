"""
app/storage/document_storage.py
---------------------------------
DocumentStorage abstraction for EWRGV corpus artifacts.

Design rationale
----------------
* Workers and services must NOT directly call SupabaseStorageClient.
  They interact with DocumentStorage instead.
* This makes the storage backend replaceable without touching business logic.
* The only concrete implementation is SupabaseDocumentStorage, which wraps
  the existing SupabaseStorageClient and uses artifact_paths for all paths.
* Future backends (S3, GCS, local filesystem) can be added by implementing
  the DocumentStorage ABC.

Status-consistency guarantee
-----------------------------
All upload operations:
  1. Perform the upload.
  2. Return the path on success.
  3. Raise StorageError on failure.
The CALLER is responsible for updating DB status ONLY after a successful
return. This is the "artifact must exist before success status" rule.

Path semantics
--------------
All paths returned by this module are BUCKET-RELATIVE (no bucket prefix).
The bucket name is encapsulated inside SupabaseDocumentStorage.
"""

from __future__ import annotations

__all__ = [
    "DocumentStorage",
    "SupabaseDocumentStorage",
]

import contextlib
from abc import ABC, abstractmethod
from typing import Optional

import httpx

from app.core.config import settings
from app.core.exceptions import StorageError
from app.core.logging import get_logger
from app.storage.artifact_paths import ARTIFACT_BUCKET, ArtifactType, get_artifact_path
from app.storage.supabase_storage import SupabaseStorageClient

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class DocumentStorage(ABC):
    """
    Abstract interface for corpus artifact storage.

    Application code (workers, services) depends only on this interface.
    Supabase, S3, GCS, or local filesystem implementations all satisfy it.

    All path arguments and return values are ARTIFACT-TYPE-BASED, not raw
    path strings.  Use ArtifactType enum values, not literal paths.

    Lifecycle
    ---------
    Implementations that hold a connection (e.g. HTTP client) must be used
    as async context managers OR have close() called explicitly.
    """

    @abstractmethod
    async def upload(
        self,
        paper_id: str,
        artifact_type: ArtifactType,
        content: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """
        Upload an artifact for the given paper.

        Parameters
        ----------
        paper_id:
            Canonical paper UUID.
        artifact_type:
            Which artifact to store.
        content:
            Raw bytes to upload.
        content_type:
            MIME type of the content.

        Returns
        -------
        str
            Bucket-relative path of the stored artifact.

        Raises
        ------
        StorageError:
            On any upload failure.
        """
        ...

    @abstractmethod
    async def download(
        self,
        paper_id: str,
        artifact_type: ArtifactType,
    ) -> bytes:
        """
        Download an artifact for the given paper.

        Returns
        -------
        bytes:
            Raw file content.

        Raises
        ------
        StorageError:
            If the artifact does not exist or download fails.
        """
        ...

    @abstractmethod
    async def exists(
        self,
        paper_id: str,
        artifact_type: ArtifactType,
    ) -> bool:
        """
        Check whether an artifact exists in storage.

        Returns
        -------
        bool:
            True if the artifact is present, False otherwise.
            Never raises on a missing file — only raises on unexpected errors.
        """
        ...

    @abstractmethod
    async def delete(
        self,
        paper_id: str,
        artifact_type: ArtifactType,
    ) -> None:
        """
        Delete an artifact from storage.

        Does NOT update any DB status — that is the caller's responsibility.

        Raises
        ------
        StorageError:
            If deletion fails for a reason other than the file being absent.
        """
        ...

    @abstractmethod
    def get_path(
        self,
        paper_id: str,
        artifact_type: ArtifactType,
    ) -> str:
        """
        Return the bucket-relative path for an artifact without performing I/O.

        Useful for persisting the path in the DB before the actual upload.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release any underlying resources (HTTP client, file handles, etc.)."""
        ...


# ---------------------------------------------------------------------------
# Supabase implementation
# ---------------------------------------------------------------------------


class SupabaseDocumentStorage(DocumentStorage):
    """
    DocumentStorage backed by Supabase Storage via REST API.

    Uses :class:`~app.storage.supabase_storage.SupabaseStorageClient`
    for all HTTP operations and
    :mod:`~app.storage.artifact_paths` for all path construction.

    Parameters
    ----------
    supabase_url:
        Supabase project URL.  Defaults to ``settings.SUPABASE_URL``.
    service_role_key:
        Service-role JWT.  Defaults to ``settings.SUPABASE_SERVICE_ROLE_KEY``.
    bucket:
        Storage bucket name.  Defaults to ``ARTIFACT_BUCKET`` (from
        ``settings.SUPABASE_STORAGE_BUCKET``).
    """

    def __init__(
        self,
        supabase_url: str | None = None,
        service_role_key: str | None = None,
        bucket: str | None = None,
    ) -> None:
        self._bucket = bucket or ARTIFACT_BUCKET
        self._client = SupabaseStorageClient(
            supabase_url=supabase_url,
            service_role_key=service_role_key,
        )

    # ------------------------------------------------------------------ #
    # DocumentStorage implementation
    # ------------------------------------------------------------------ #

    def get_path(self, paper_id: str, artifact_type: ArtifactType) -> str:
        """Return the bucket-relative artifact path (no I/O)."""
        return get_artifact_path(paper_id, artifact_type)

    async def upload(
        self,
        paper_id: str,
        artifact_type: ArtifactType,
        content: bytes,
        content_type: str = "application/octet-stream",
    ) -> str:
        """
        Upload an artifact.  Returns the bucket-relative path on success.

        The caller MUST update the DB status only after this returns without
        raising an exception.
        """
        path = get_artifact_path(paper_id, artifact_type)
        logger.debug(
            "DocumentStorage.upload",
            extra={"paper_id": paper_id, "artifact_type": artifact_type, "path": path},
        )
        await self._client.upload(
            bucket=self._bucket,
            path=path,
            content=content,
            content_type=content_type,
            upsert=True,  # Idempotent: second upload overwrites cleanly
        )
        return path

    async def download(self, paper_id: str, artifact_type: ArtifactType) -> bytes:
        """Download an artifact.  Raises StorageError if missing."""
        path = get_artifact_path(paper_id, artifact_type)
        logger.debug(
            "DocumentStorage.download",
            extra={"paper_id": paper_id, "artifact_type": artifact_type, "path": path},
        )
        return await self._client.download(bucket=self._bucket, path=path)

    async def exists(self, paper_id: str, artifact_type: ArtifactType) -> bool:
        """
        Check artifact existence via a HEAD request.

        Returns False (not raises) if the file is absent.
        Raises StorageError only for unexpected errors.
        """
        path = get_artifact_path(paper_id, artifact_type)
        url = (
            f"{self._client._url}/storage/v1/object/{self._bucket}/{path}"
        )
        headers = self._client._auth_headers()

        try:
            response = await self._client._client.head(url, headers=headers)
        except httpx.RequestError as exc:
            raise StorageError(
                f"Storage existence check network error: {exc}",
                detail=f"bucket={self._bucket} path={path}",
            ) from exc

        if response.status_code == 200:
            return True
        if response.status_code in (404, 400):
            return False
        # Unexpected status — treat as storage error
        raise StorageError(
            f"Storage existence check unexpected HTTP {response.status_code}",
            detail=f"bucket={self._bucket} path={path}",
        )

    async def delete(self, paper_id: str, artifact_type: ArtifactType) -> None:
        """
        Delete an artifact.

        Safe to call even if the file does not exist (no-op on 404).
        Raises StorageError for unexpected failures.
        """
        path = get_artifact_path(paper_id, artifact_type)
        url = (
            f"{self._client._url}/storage/v1/object/{self._bucket}/{path}"
        )
        headers = self._client._auth_headers()

        try:
            response = await self._client._client.delete(url, headers=headers)
        except httpx.RequestError as exc:
            raise StorageError(
                f"Storage delete network error: {exc}",
                detail=f"bucket={self._bucket} path={path}",
            ) from exc

        # 200 = deleted, 404 = already absent (idempotent)
        if response.status_code not in (200, 404):
            raise StorageError(
                f"Storage delete failed: HTTP {response.status_code}",
                detail=f"bucket={self._bucket} path={path}",
            )

        logger.debug(
            "DocumentStorage.delete",
            extra={"paper_id": paper_id, "artifact_type": artifact_type, "path": path},
        )

    async def close(self) -> None:
        """Release the underlying HTTP client."""
        await self._client.close()

    # ------------------------------------------------------------------ #
    # Context manager support
    # ------------------------------------------------------------------ #

    async def __aenter__(self) -> "SupabaseDocumentStorage":
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.close()
