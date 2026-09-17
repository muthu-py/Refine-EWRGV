"""
app/storage/supabase_storage.py
--------------------------------
Shared Supabase Storage REST API client for EWRGV.

Provides upload and download operations against a Supabase Storage bucket
via the Supabase Storage REST API.  This client is used by Phase 2.5
(document parsing) to download acquired PDFs and upload parsed.json files.

Design notes
------------
* This is an additive module.  The existing fulltext.py (Phase 2.4) is
  NOT modified; it continues to use its own inline httpx calls.
* A single httpx.AsyncClient is shared per SupabaseStorageClient instance.
  Callers are responsible for calling close() when finished.
* No Supabase credentials are logged.
* Raises StorageError (from app.core.exceptions) on failure so the caller
  can decide how to handle the error.
"""

from __future__ import annotations

import httpx

from app.core.config import settings
from app.core.exceptions import StorageError
from app.core.logging import get_logger

logger = get_logger(__name__)

# Maximum download size accepted by the parser (20 MB)
_MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024


class SupabaseStorageClient:
    """
    Async client for Supabase Storage REST API.

    Parameters
    ----------
    supabase_url:
        Base URL of the Supabase project (e.g. https://xyz.supabase.co).
        Defaults to settings.SUPABASE_URL.
    service_role_key:
        Service-role JWT for authenticated access.
        Defaults to settings.SUPABASE_SERVICE_ROLE_KEY.
    """

    def __init__(
        self,
        supabase_url: str | None = None,
        service_role_key: str | None = None,
    ) -> None:
        self._url = (supabase_url or settings.SUPABASE_URL or "").rstrip("/")
        self._key = service_role_key or settings.SUPABASE_SERVICE_ROLE_KEY or ""
        self._client = httpx.AsyncClient(
            timeout=60.0,
            follow_redirects=True,
        )

    async def close(self) -> None:
        """Release the underlying HTTP client."""
        await self._client.aclose()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def download(self, bucket: str, path: str) -> bytes:
        """
        Download a file from Supabase Storage.

        Parameters
        ----------
        bucket:
            Storage bucket name (e.g. "papers").
        path:
            Bucket-relative path to the file (e.g. "papers/{id}/fulltext.pdf").
            Must NOT include the bucket name as a prefix.

        Returns
        -------
        bytes:
            Raw file content.

        Raises
        ------
        StorageError:
            On HTTP error, empty response, or oversized file.
        """
        self._require_credentials()
        url = f"{self._url}/storage/v1/object/{bucket}/{path}"
        headers = self._auth_headers()

        try:
            response = await self._client.get(url, headers=headers)
        except httpx.RequestError as exc:
            raise StorageError(
                f"Storage download network error: {exc}",
                detail=f"bucket={bucket} path={path}",
            ) from exc

        if response.status_code != 200:
            raise StorageError(
                f"Storage download failed: HTTP {response.status_code}",
                detail=f"bucket={bucket} path={path}",
            )

        content = response.content
        if not content:
            raise StorageError(
                "Storage download returned empty content",
                detail=f"bucket={bucket} path={path}",
            )

        if len(content) > _MAX_DOWNLOAD_BYTES:
            raise StorageError(
                f"Storage file too large: {len(content)} bytes",
                detail=f"bucket={bucket} path={path} limit={_MAX_DOWNLOAD_BYTES}",
            )

        return content

    async def upload(
        self,
        bucket: str,
        path: str,
        content: bytes,
        content_type: str = "application/octet-stream",
        upsert: bool = True,
    ) -> str:
        """
        Upload a file to Supabase Storage.

        Parameters
        ----------
        bucket:
            Storage bucket name (e.g. "papers").
        path:
            Bucket-relative destination path (e.g. "papers/{id}/parsed.json").
        content:
            Raw bytes to upload.
        content_type:
            MIME type of the file being uploaded.
        upsert:
            If True, overwrite an existing file at the same path.

        Returns
        -------
        str:
            The bucket-relative storage path of the uploaded file (i.e. ``path``).

        Raises
        ------
        StorageError:
            On HTTP error or network failure.
        """
        self._require_credentials()
        url = f"{self._url}/storage/v1/object/{bucket}/{path}"
        headers = {
            **self._auth_headers(),
            "Content-Type": content_type,
            "x-upsert": "true" if upsert else "false",
        }

        try:
            response = await self._client.post(url, content=content, headers=headers)
        except httpx.RequestError as exc:
            raise StorageError(
                f"Storage upload network error: {exc}",
                detail=f"bucket={bucket} path={path}",
            ) from exc

        if response.status_code not in (200, 201):
            raise StorageError(
                f"Storage upload failed: HTTP {response.status_code}",
                detail=f"bucket={bucket} path={path}",
            )

        return path

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._key}",
            "apikey": self._key,
        }

    def _require_credentials(self) -> None:
        if not self._url or not self._key:
            raise StorageError(
                "Supabase Storage credentials are not configured. "
                "Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY."
            )
