"""
app/providers/search/semantic_scholar.py
-----------------------------------------
Semantic Scholar literature collector.

Implements the LiteratureCollector interface using the Semantic Scholar API.
Status: SKELETON – not yet implemented.
"""

from __future__ import annotations

from typing import Optional

from app.core.logging import get_logger
from app.domain.models.paper import Paper

logger = get_logger(__name__)


class SemanticScholarCollector:
    """
    LiteratureCollector backed by the Semantic Scholar Academic Graph API.

    Status: SKELETON – not yet implemented.
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: str = "") -> None:
        self._api_key = api_key

    def search(self, query: str, limit: int = 20) -> list[Paper]:
        """Search for papers. NOT YET IMPLEMENTED."""
        logger.warning("SemanticScholarCollector.search is not yet implemented.")
        return []

    def fetch_by_id(self, paper_id: str) -> Optional[Paper]:
        """Fetch a paper by Semantic Scholar ID. NOT YET IMPLEMENTED."""
        logger.warning("SemanticScholarCollector.fetch_by_id is not yet implemented.")
        return None
