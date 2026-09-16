"""
app/ingestion/collectors/service.py
-------------------------------------
LiteratureCollectionService — Phase 2.2 core business logic.

Responsibilities
----------------
1. Accept a ResearchQuery (with expanded_queries from Phase 2.1).
2. Dispatch each expanded query to all configured provider adapters.
3. Accumulate CollectedPaper objects with full provenance.
4. Record provider failures independently — a failure in one provider
   does not abort the rest of the collection.
5. Return a CollectionResult ready for Phase 2.3 deduplication.

Architecture position
---------------------
    ResearchPipeline / API route
              ↓
    LiteratureCollectionService   ← this file
              ↓
    SemanticScholarCollector | OpenAlexCollector   (provider adapters)
              ↓
    CollectedPaper objects (Paper + PaperProvenance)

Error handling
--------------
* IngestionError from a single (provider, query) pair is caught and
  recorded in CollectionResult.provider_errors.  Other queries/providers
  continue.
* If ALL providers fail for ALL queries, status = "failed".
* If some succeed and some fail, status = "partial".
* If all succeed, status = "success".

Design principles
-----------------
* The service depends on the provider *instances* injected at
  construction time, not on concrete classes.  Tests can inject mocks.
* No deduplication is performed here.  Duplicate papers across
  providers/queries are preserved for Phase 2.3.
* No PDFs are fetched; no embeddings are generated.
"""

from __future__ import annotations

from typing import Any

from app.core.exceptions import IngestionError
from app.core.logging import get_logger
from app.domain.models.collection import CollectedPaper, CollectionResult
from app.domain.models.query import ResearchQuery

logger = get_logger(__name__)


class LiteratureCollectionService:
    """
    Orchestrates literature collection across multiple provider adapters.

    Parameters
    ----------
    providers:
        Mapping of provider name → collector instance.  Each collector
        must implement ``search(query: str, limit: int) -> list[CollectedPaper]``.
    max_results_per_query:
        Maximum papers to request from each provider per query.
    max_total_results:
        Hard cap on total collected papers (across all providers + queries).
        Collection stops once this limit is reached.
    """

    def __init__(
        self,
        providers: dict[str, Any],
        max_results_per_query: int = 20,
        max_total_results: int = 200,
    ) -> None:
        if not providers:
            raise ValueError("LiteratureCollectionService requires at least one provider.")
        self._providers = providers
        self._max_results_per_query = max_results_per_query
        self._max_total_results = max_total_results

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def collect(self, research_query: ResearchQuery) -> CollectionResult:
        """
        Run literature collection for the given ResearchQuery.

        Uses ``research_query.expanded_queries`` as the primary search
        input.  Falls back to ``[research_query.query]`` if
        ``expanded_queries`` is empty.

        Parameters
        ----------
        research_query:
            Enriched ResearchQuery from Phase 2.1.

        Returns
        -------
        CollectionResult
            All collected papers + provenance + any provider errors.
        """
        queries = research_query.expanded_queries or [research_query.query]

        logger.info(
            "LiteratureCollectionService.collect: starting",
            extra={
                "query_id": research_query.query_id,
                "queries_count": len(queries),
                "providers": list(self._providers.keys()),
            },
        )

        all_papers: list[CollectedPaper] = []
        provider_errors: list[dict] = []
        providers_used: list[str] = []
        total_cap_reached = False

        for provider_name, collector in self._providers.items():
            if total_cap_reached:
                break

            provider_contributed = False
            for query in queries:
                if total_cap_reached:
                    break

                remaining = self._max_total_results - len(all_papers)
                limit = min(self._max_results_per_query, remaining)
                if limit <= 0:
                    total_cap_reached = True
                    logger.info(
                        "LiteratureCollectionService: max_total_results reached; stopping.",
                        extra={"total": len(all_papers)},
                    )
                    break

                try:
                    papers = collector.search(query, limit=limit)
                    all_papers.extend(papers)
                    provider_contributed = True
                    logger.debug(
                        "LiteratureCollectionService: query result",
                        extra={
                            "provider": provider_name,
                            "query": query[:60],
                            "count": len(papers),
                        },
                    )
                except IngestionError as exc:
                    logger.warning(
                        "LiteratureCollectionService: provider error",
                        extra={
                            "provider": provider_name,
                            "query": query[:60],
                            "error": str(exc),
                        },
                    )
                    provider_errors.append({
                        "provider": provider_name,
                        "query": query,
                        "error": str(exc),
                    })
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "LiteratureCollectionService: unexpected error",
                        extra={
                            "provider": provider_name,
                            "query": query[:60],
                            "error": str(exc),
                        },
                    )
                    provider_errors.append({
                        "provider": provider_name,
                        "query": query,
                        "error": f"Unexpected error: {exc}",
                    })

            if provider_contributed and provider_name not in providers_used:
                providers_used.append(provider_name)

        # Determine overall status
        if all_papers:
            status = "partial" if provider_errors else "success"
        else:
            status = "failed"

        result = CollectionResult(
            query_id=research_query.query_id,
            queries_searched=queries,
            providers_used=providers_used,
            papers=all_papers,
            provider_errors=provider_errors,
            status=status,
        )

        logger.info(
            "LiteratureCollectionService.collect: complete",
            extra={
                "query_id": research_query.query_id,
                "status": status,
                "total": len(all_papers),
                "errors": len(provider_errors),
            },
        )
        return result
