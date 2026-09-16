"""
app/providers/search/semantic_scholar.py
-----------------------------------------
Semantic Scholar literature collector.

Implements the LiteratureCollector interface using the Semantic Scholar
Academic Graph (S2AG) API v1.

API reference: https://api.semanticscholar.org/graph/v1

Design notes
------------
* Uses the ``GET /paper/search`` endpoint.
* Requests a specific field list to avoid fetching unnecessary data.
* API key is optional — anonymous access is permitted but rate-limited.
* All provider-specific JSON is contained here; downstream code sees only
  canonical ``Paper`` + ``PaperProvenance`` objects.
* The normaliser maps every field defensively; missing fields become None
  or empty defaults — no invented metadata.
"""

from __future__ import annotations

from typing import Any, Optional

from app.core.exceptions import IngestionError
from app.core.logging import get_logger
from app.domain.models.collection import CollectedPaper, PaperProvenance
from app.domain.models.paper import Author, Paper
from app.ingestion.collectors.http_client import LiteratureHttpClient

logger = get_logger(__name__)

# Fields requested from the Semantic Scholar API
_S2_FIELDS = ",".join([
    "paperId",
    "title",
    "authors",
    "abstract",
    "year",
    "venue",
    "externalIds",
    "openAccessPdf",
    "citationCount",
    "url",
])

_SOURCE = "semantic_scholar"


class SemanticScholarCollector:
    """
    LiteratureCollector backed by the Semantic Scholar Academic Graph API.

    Parameters
    ----------
    api_key:
        Optional S2AG API key.  Anonymous access is allowed but more
        heavily rate-limited.
    base_url:
        API base URL.
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.semanticscholar.org/graph/v1",
        timeout: int = 30,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout

        headers: dict[str, str] = {}
        if api_key:
            headers["x-api-key"] = api_key

        self._http = LiteratureHttpClient(
            base_url=base_url,
            timeout=timeout,
            default_headers=headers,
        )

    # ------------------------------------------------------------------ #
    # Public interface (satisfies LiteratureCollector protocol)
    # ------------------------------------------------------------------ #

    def search(
        self, query: str, limit: int = 20
    ) -> list[CollectedPaper]:
        """
        Search Semantic Scholar for papers matching ``query``.

        Parameters
        ----------
        query:
            Free-text search query.
        limit:
            Maximum number of results to return (capped by the API).

        Returns
        -------
        list[CollectedPaper]
            Normalised papers with provenance.

        Raises
        ------
        IngestionError
            On HTTP errors, rate limits, or malformed responses.
        """
        logger.info(
            "SemanticScholarCollector.search",
            extra={"query": query[:80], "limit": limit},
        )
        try:
            raw = self._http.get(
                "/paper/search",
                params={"query": query, "limit": limit, "fields": _S2_FIELDS},
            )
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(
                f"Semantic Scholar search failed: {exc}",
                detail=str(exc),
            ) from exc

        if not isinstance(raw, dict):
            raise IngestionError(
                "Semantic Scholar returned unexpected response type.",
                detail=f"Expected dict, got {type(raw).__name__}",
            )

        data = raw.get("data", [])
        if not isinstance(data, list):
            raise IngestionError(
                "Semantic Scholar 'data' field is not a list.",
                detail=str(type(data)),
            )

        results: list[CollectedPaper] = []
        for item in data:
            try:
                cp = self._normalise(item, source_query=query)
                results.append(cp)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "SemanticScholarCollector: skipping malformed item",
                    extra={"error": str(exc)},
                )

        logger.info(
            "SemanticScholarCollector.search complete",
            extra={"query": query[:80], "results": len(results)},
        )
        return results

    def fetch_by_id(self, paper_id: str) -> Optional[CollectedPaper]:
        """
        Fetch a single paper by its Semantic Scholar paper ID.

        Returns None if the paper is not found.
        """
        logger.info("SemanticScholarCollector.fetch_by_id", extra={"paper_id": paper_id})
        try:
            raw = self._http.get(
                f"/paper/{paper_id}",
                params={"fields": _S2_FIELDS},
            )
        except IngestionError as exc:
            if "HTTP 404" in str(exc):
                return None
            raise

        if not isinstance(raw, dict):
            return None

        try:
            return self._normalise(raw, source_query="")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "SemanticScholarCollector.fetch_by_id: normalisation failed",
                extra={"paper_id": paper_id, "error": str(exc)},
            )
            return None

    # ------------------------------------------------------------------ #
    # Normalisation
    # ------------------------------------------------------------------ #

    def _normalise(self, item: dict[str, Any], source_query: str) -> CollectedPaper:
        """
        Map a Semantic Scholar API response item to ``CollectedPaper``.

        All field accesses are defensive: missing or None values produce
        safe defaults rather than raising exceptions.
        """
        provider_id: str = item.get("paperId") or ""
        title: str = (item.get("title") or "").strip()
        if not title:
            title = "(No title)"

        # Authors
        authors: list[Author] = []
        for a in item.get("authors") or []:
            name = (a.get("name") or "").strip()
            if name:
                authors.append(Author(name=name))

        abstract: str = (item.get("abstract") or "").strip()
        publication_year: Optional[int] = item.get("year")

        venue: Optional[str] = item.get("venue") or None
        if venue:
            venue = venue.strip() or None

        # DOI from externalIds
        doi: Optional[str] = None
        external_ids = item.get("externalIds") or {}
        if isinstance(external_ids, dict):
            doi = external_ids.get("DOI") or None

        # Open Access PDF
        full_text_url: Optional[str] = None
        open_access_status: Optional[str] = None
        oa_pdf = item.get("openAccessPdf") or {}
        if isinstance(oa_pdf, dict):
            full_text_url = oa_pdf.get("url") or None
            open_access_status = oa_pdf.get("status") or None

        citation_count: Optional[int] = item.get("citationCount")
        paper_url: Optional[str] = item.get("url") or None

        paper = Paper(
            title=title,
            authors=authors,
            abstract=abstract,
            source=_SOURCE,
            venue=venue,
            doi=doi,
            provider_id=provider_id,
            paper_url=paper_url,
            full_text_url=full_text_url,
            open_access_status=open_access_status,
            citation_count=citation_count,
            publication_year=publication_year,
        )
        provenance = PaperProvenance(
            source=_SOURCE,
            provider_id=provider_id,
            source_query=source_query,
        )
        return CollectedPaper(paper=paper, provenance=provenance)
