"""
app/ingestion/acquisition/fulltext.py
-------------------------------------
Downloads full-text PDFs and uploads them to Supabase Storage.
"""

import httpx
from typing import Optional, Dict, Any

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class FullTextAcquisitionService:
    """
    Acquires Open Access full-text documents and uploads them directly to Supabase Storage.
    """
    def __init__(self):
        self._supabase_url = settings.SUPABASE_URL.rstrip('/') if settings.SUPABASE_URL else ""
        self._supabase_key = settings.SUPABASE_SERVICE_ROLE_KEY
        self._bucket = settings.SUPABASE_STORAGE_BUCKET
        self._client = httpx.AsyncClient(timeout=30.0, follow_redirects=True)
        self._max_file_size = 20 * 1024 * 1024  # 20 MB max

    async def close(self):
        await self._client.aclose()

    async def acquire(self, paper_id: str, full_text_url: str) -> dict:
        """
        Attempts to download the PDF and upload it to Supabase storage.
        Returns a dict with the outcome:
        {
            "status": "DOWNLOADED" | "DOWNLOAD_FAILED" | "INVALID_DOCUMENT",
            "storage_path": str | None,
            "content_type": str | None,
            "file_size": int | None
        }
        """
        if not self._supabase_url or not self._supabase_key:
            logger.error("Supabase Storage credentials not configured.")
            return {"status": "DOWNLOAD_FAILED", "storage_path": None, "content_type": None, "file_size": None}

        try:
            # 1. Download the PDF
            # Use stream to avoid loading huge non-PDF files into memory
            async with self._client.stream("GET", full_text_url) as response:
                if response.status_code != 200:
                    return {"status": "DOWNLOAD_FAILED", "storage_path": None, "content_type": None, "file_size": None}

                content_type = response.headers.get("Content-Type", "").lower()
                
                # Check strict PDF content type
                if "application/pdf" not in content_type:
                    return {"status": "INVALID_DOCUMENT", "storage_path": None, "content_type": content_type, "file_size": None}
                
                content_length = int(response.headers.get("Content-Length", 0))
                if content_length > self._max_file_size:
                    return {"status": "DOWNLOAD_FAILED", "storage_path": None, "content_type": content_type, "file_size": content_length}

                # Read content
                file_bytes = await response.aread()
                file_size = len(file_bytes)

                if file_size == 0 or file_size > self._max_file_size:
                    return {"status": "DOWNLOAD_FAILED", "storage_path": None, "content_type": content_type, "file_size": file_size}

                # 2. Upload to Supabase Storage REST API
                storage_path = f"papers/{paper_id}/fulltext.pdf"
                upload_url = f"{self._supabase_url}/storage/v1/object/{self._bucket}/{storage_path}"
                
                upload_headers = {
                    "Authorization": f"Bearer {self._supabase_key}",
                    "Content-Type": "application/pdf",
                    "x-upsert": "true"
                }
                
                upload_resp = await self._client.post(
                    upload_url,
                    content=file_bytes,
                    headers=upload_headers
                )

                if upload_resp.status_code not in (200, 201):
                    logger.error(
                        "Supabase storage upload failed", 
                        extra={"status_code": upload_resp.status_code, "resp": upload_resp.text}
                    )
                    return {"status": "DOWNLOAD_FAILED", "storage_path": None, "content_type": content_type, "file_size": file_size}

                return {
                    "status": "DOWNLOADED",
                    "storage_path": f"{self._bucket}/{storage_path}",
                    "content_type": "application/pdf",
                    "file_size": file_size
                }
                
        except Exception as exc:
            logger.warning(
                "Full-text acquisition failed",
                extra={"paper_id": paper_id, "url": full_text_url, "error": str(exc)}
            )
            return {"status": "DOWNLOAD_FAILED", "storage_path": None, "content_type": None, "file_size": None}
