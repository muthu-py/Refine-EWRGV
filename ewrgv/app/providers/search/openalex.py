"""
app/providers/search/openalex.py
---------------------------------
OpenAlex literature collector.

Implements the LiteratureCollector interface using the OpenAlex API.

API reference: https://docs.openalex.org/api-entities/works/search-works

Design notes
------------
* Uses the ``GET /works?search=<query>`` endpoint.
* No API key is required.  Providing an email via the ``mailto`` parameter
  opts in to the "polite pool" with higher rate limits.
* Abstract text is reconstructed from the inverted-index format that
  OpenAlex returns (``abstract_inverted_index``).
* All provider-specific JSON is encapsulated here; downstream code sees
  only canonical ``Paper`` + ``PaperProvenance`` objects.
* Missing fields produce None / empty defaults — no invented metadata.
"""

from __future__ import annotations

from typing import Any, Optional

from app.core.exceptions import IngestionError
from app.core.logging import get_logger
from app.domain.models.collection import CollectedPaper, PaperProvenance
from app.domain.models.paper import Author, Paper
from app.ingestion.collectors.http_client import LiteratureHttpClient

logger = get_logger(__name__)

_SOURCE = "openalex"


def _reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """
    Reconstruct abstract text from OpenAlex abstract_inverted_index.

    OpenAlex stores abstracts as ``{word: [position, ...], ...}``.
    This function reverses the index into the original word sequence.

    Returns an empty string if the index is absent or malformed.
    """
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    try:
        positions: dict[int, str] = {}
        for word, pos_list in inverted_index.items():
            for pos in pos_list:
                positions[pos] = word
        if not positions:
            return ""
        return " ".join(positions[i] for i in sorted(positions))
    except Exception:  # noqa: BLE001
        return ""


class OpenAlexCollector:
    """
    LiteratureCollector backed by the OpenAlex API.

    Parameters
    ----------
    base_url:
        OpenAlex API base URL.
    mailto:
        Email address for the polite pool (strongly recommended).
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "https://api.openalex.org",
        mailto: str = "",
        timeout: int = 30,
    ) -> None:
        self._base_url = base_url
        self._mailto = mailto
        self._timeout = timeout
        self._http = LiteratureHttpClient(
            base_url=base_url,
            timeout=timeout,
            default_headers={"User-Agent": f"EWRGV/0.1 (mailto:{mailto})" if mailto else "EWRGV/0.1"},
        )

    # ------------------------------------------------------------------ #
    # Public interface (satisfies LiteratureCollector protocol)
    # ------------------------------------------------------------------ #

    def search(
        self, query: str, limit: int = 20
    ) -> list[CollectedPaper]:
        """
        Search OpenAlex for works matching ``query``.

        Parameters
        ----------
        query:
            Free-text search query.
        limit:
            Maximum number of results to return.

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
            "OpenAlexCollector.search",
            extra={"query": query[:80], "limit": limit},
        )
        params: dict[str, Any] = {
            "search": query,
            "per-page": min(limit, 200),  # OpenAlex max per-page is 200
            "select": "id,title,authorships,abstract_inverted_index,publication_year,"
                      "primary_location,doi,open_access,cited_by_count,type",
        }
        if self._mailto:
            params["mailto"] = self._mailto

        try:
            raw = self._http.get("/works", params=params)
        except IngestionError:
            raise
        except Exception as exc:
            raise IngestionError(
                f"OpenAlex search failed: {exc}",
                detail=str(exc),
            ) from exc

        if not isinstance(raw, dict):
            raise IngestionError(
                "OpenAlex returned unexpected response type.",
                detail=f"Expected dict, got {type(raw).__name__}",
            )

        results_list = raw.get("results", [])
        if not isinstance(results_list, list):
            raise IngestionError(
                "OpenAlex 'results' field is not a list.",
                detail=str(type(results_list)),
            )

        results: list[CollectedPaper] = []
        for item in results_list:
            try:
                cp = self._normalise(item, source_query=query)
                results.append(cp)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "OpenAlexCollector: skipping malformed item",
                    extra={"error": str(exc)},
                )

        logger.info(
            "OpenAlexCollector.search complete",
            extra={"query": query[:80], "results": len(results)},
        )
        return results

    def fetch_by_id(self, work_id: str) -> Optional[CollectedPaper]:
        """
        Fetch a single work by its OpenAlex work ID (e.g. W2741809807).

        Returns None if not found.
        """
        logger.info("OpenAlexCollector.fetch_by_id", extra={"work_id": work_id})
        params: dict[str, Any] = {}
        if self._mailto:
            params["mailto"] = self._mailto
        try:
            raw = self._http.get(f"/works/{work_id}", params=params)
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
                "OpenAlexCollector.fetch_by_id: normalisation failed",
                extra={"work_id": work_id, "error": str(exc)},
            )
            return None

    # ------------------------------------------------------------------ #
    # Normalisation
    # ------------------------------------------------------------------ #

    def _normalise(self, item: dict[str, Any], source_query: str) -> CollectedPaper:
        """
        Map an OpenAlex work object to ``CollectedPaper``.

        All field accesses are defensive.
        """
        provider_id: str = (item.get("id") or "").strip()
        # OpenAlex IDs look like "https://openalex.org/W123456"; strip to short form
        short_id = provider_id.replace("https://openalex.org/", "") or provider_id

        title: str = (item.get("title") or "").strip()
        if not title:
            title = "(No title)"

        # Authors from authorships
        authors: list[Author] = []
        for authorship in item.get("authorships") or []:
            author_obj = authorship.get("author") or {}
            name = (author_obj.get("display_name") or "").strip()
            if name:
                authors.append(Author(name=name))

        # Abstract from inverted index
        abstract = _reconstruct_abstract(item.get("abstract_inverted_index"))

        publication_year: Optional[int] = item.get("publication_year")

        # Venue from primary_location → source
        venue: Optional[str] = None
        primary_location = item.get("primary_location") or {}
        if isinstance(primary_location, dict):
            source_obj = primary_location.get("source") or {}
            if isinstance(source_obj, dict):
                venue = (source_obj.get("display_name") or "").strip() or None

        # DOI
        doi_raw = item.get("doi") or None
        doi: Optional[str] = None
        if doi_raw:
            doi = str(doi_raw).replace("https://doi.org/", "").strip() or None

        # Open Access
        oa = item.get("open_access") or {}
        open_access_status: Optional[str] = None
        full_text_url: Optional[str] = None
        if isinstance(oa, dict):
            open_access_status = oa.get("oa_status") or None
            full_text_url = oa.get("oa_url") or None

        citation_count: Optional[int] = item.get("cited_by_count")

        # Canonical URL
        paper_url: Optional[str] = provider_id if provider_id.startswith("http") else None

        paper = Paper(
            title=title,
            authors=authors,
            abstract=abstract,
            source=_SOURCE,
            venue=venue,
            doi=doi,
            provider_id=short_id,
            paper_url=paper_url,
            full_text_url=full_text_url,
            open_access_status=open_access_status,
            citation_count=citation_count,
            publication_year=publication_year,
        )
        provenance = PaperProvenance(
            source=_SOURCE,
            provider_id=short_id,
            source_query=source_query,
        )
        return CollectedPaper(paper=paper, provenance=provenance)
