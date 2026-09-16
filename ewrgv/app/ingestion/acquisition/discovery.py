"""
app/ingestion/acquisition/discovery.py
--------------------------------------
Discovers Open Access full-text URLs for papers via Unpaywall.
"""

import httpx
from typing import Optional, Dict, Any

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

class OADiscoveryService:
    """
    Checks for Open Access links.
    If the paper already has a `full_text_url` from OpenAlex/S2, it trusts it.
    Otherwise, checks Unpaywall if a DOI is present.
    """
    def __init__(self):
        self._unpaywall_url = settings.UNPAYWALL_BASE_URL
        self._email = settings.UNPAYWALL_EMAIL
        self._client = httpx.AsyncClient(timeout=15.0)

    async def close(self):
        await self._client.aclose()

    async def discover(self, paper: dict) -> tuple[Optional[str], Optional[str]]:
        """
        Takes a paper dictionary (from the DB) and returns a tuple:
        (best_oa_url, source_type)
        
        If it already has an OA URL, returns it immediately.
        If not, falls back to Unpaywall using the DOI.
        Returns (None, None) if no OA link can be found.
        """
        # 1. Check if we already have one from collection (OpenAlex/Semantic Scholar)
        if paper.get("full_text_url"):
            # It was populated during collection. The source type was stored in `source_type`.
            return paper["full_text_url"], paper.get("source_type")

        # 2. Check Unpaywall via DOI
        doi = paper.get("doi")
        if not doi or not self._email:
            return None, None
            
        return await self._check_unpaywall(doi)

    async def _check_unpaywall(self, doi: str) -> tuple[Optional[str], Optional[str]]:
        """Queries the Unpaywall API using the paper's DOI."""
        url = f"{self._unpaywall_url}/{doi}"
        params = {"email": self._email}
        
        try:
            response = await self._client.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                if data.get("is_oa") and data.get("best_oa_location"):
                    best_location = data["best_oa_location"]
                    oa_url = best_location.get("url_for_pdf") or best_location.get("url")
                    if oa_url:
                        return oa_url, "unpaywall"
        except Exception as exc:
            logger.warning(
                "Unpaywall discovery failed",
                extra={"doi": doi, "error": str(exc)}
            )
            
        return None, None
