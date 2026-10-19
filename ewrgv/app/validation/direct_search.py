"""
app/validation/direct_search.py
---------------------------------
Validation searches.  ``ValidationSearch`` runs one query against the
configured sources and returns hits; the four EWRGV searches
(direct, terminology, method_variation, counter) are thin subclasses that
only differ in their ``search_type`` label (their queries differ, see
query_generation.py).

Sources
-------
corpus            the run's chunks (hybrid retrieval, references/boilerplate excluded)
openalex          OpenAlex relevance search (titles + reconstructed abstracts)
semantic_scholar  Semantic Scholar paper search

A hit is a dict:
    {source, search_type, query, rank, id, paper_id, title, text, year, doi, url, section, section_type, pages}
"""

from __future__ import annotations

from typing import Any, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)

ALL_SOURCES = ("corpus", "openalex", "semantic_scholar")


class ValidationSearch:
    search_type = "direct"

    def __init__(self, corpus: Any = None, retrieval: Any = None, external: Any = None,
                 sources: tuple[str, ...] = ALL_SOURCES, per_query: int = 10) -> None:
        self._corpus = corpus
        self._retrieval = retrieval
        self._external = external
        self._sources = sources
        self._per_query = per_query
        self.unavailable: set[str] = set()

    async def search(self, gap: dict, queries: list[str]) -> list[dict]:
        hits: list[dict] = []
        for q in queries:
            hits += await self.run_query(q)
        return hits

    async def run_query(self, query: str) -> list[dict]:
        out: list[dict] = []
        if "corpus" in self._sources and self._retrieval is not None:
            out += await self._corpus_hits(query)
        if self._external is not None:
            if "openalex" in self._sources:
                items = await self._external.openalex_search(query, self._per_query)
                out += self._external_hits("openalex", query, items)
            if "semantic_scholar" in self._sources:
                items = await self._external.s2_search(query, self._per_query)
                out += self._external_hits("semantic_scholar", query, items)
        return out

    async def _corpus_hits(self, query: str) -> list[dict]:
        try:
            results = await self._retrieval.aretrieve_many([query], top_k=min(self._per_query, 10), method="hybrid")
        except Exception as exc:  # noqa: BLE001 - one failed query must not stop validation
            logger.warning("corpus search failed for %r: %s", query[:60], exc)
            return []
        hits = []
        for r in results:
            meta = r.metadata or {}
            hits.append({
                "source": "corpus", "search_type": self.search_type, "query": query, "rank": r.rank,
                "id": r.chunk_id, "paper_id": r.paper_id, "title": meta.get("title") or "",
                "text": r.text, "year": meta.get("publication_year"), "doi": meta.get("doi"), "url": None,
                "section": r.section, "section_type": r.section_type,
                "pages": f"{r.page_start}-{r.page_end}" if r.page_end and r.page_end != r.page_start else (str(r.page_start) if r.page_start else None),
            })
        return hits

    def _external_hits(self, source: str, query: str, items: Optional[list]) -> list[dict]:
        if items is None:
            self.unavailable.add(source)
            return []
        hits = []
        for rank, it in enumerate(items, start=1):
            if not (it.title or it.abstract):
                continue
            hits.append({
                "source": source, "search_type": self.search_type, "query": query, "rank": rank,
                "id": it.item_id, "paper_id": None, "title": it.title, "text": it.text, "year": it.year,
                "doi": it.doi, "url": it.url, "section": None, "section_type": "ABSTRACT", "pages": None,
            })
        return hits


class DirectEvidenceSearch(ValidationSearch):
    """Queries that directly state the gap's concepts."""
    search_type = "direct"
