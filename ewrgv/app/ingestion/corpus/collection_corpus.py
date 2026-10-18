"""
app/ingestion/corpus/collection_corpus.py
-------------------------------------------
CollectionCorpusStore — an in-memory CorpusStore built from the papers
collected for a research run (Phase 2.2 CollectionResult).

Each unique paper with a non-empty abstract becomes ONE DocumentChunk whose
text is the title followed by the abstract.  Full-text (PDF) chunks are not
produced here; that requires PDF extraction, sectioning and chunking.

Satisfies the CorpusStore protocol (app.domain.interfaces), so the existing
retrievers (e.g. SparseRetriever / BM25) work on it unchanged.

Deduplication
-------------
A CollectionResult contains one CollectedPaper per (provider, query) hit,
so the same paper often appears several times.  Duplicates are merged with
the same signatures and connected-component grouping as the database
Deduplicator (app.ingestion.deduplication.service):
    * ``doi:<normalised DOI>``
    * ``tay:<normalised title | author words | year>``
    * ``prov:<provider>:<provider ID>``
Records sharing ANY signature are linked, and linked records are merged
transitively (A–B by DOI and B–C by provider ID puts A, B, C together).

Within a merged group the representative is the record with the longest
abstract (first seen on ties).  As in the database deduplicator, missing
fields are filled from the other records in collection order: DOI,
publication year, paper URL, venue, authors, and title (when the
representative's title is blank).  Provenance (providers, provider IDs,
source queries) from every record is kept in chunk metadata.  If records
with different DOIs are merged (e.g. via title + authors + year), only the
representative's DOI is kept.

Known limitation (shared with the database deduplicator): records without a
DOI or common provider ID are only linked when title, author words and year
match exactly, so author-name format differences ("A. Kumar" vs
"Arun Kumar") or preprint vs published years keep them apart.

Missing data and ID collisions
------------------------------
Papers whose abstract is missing or blank get no chunk (a title alone is
too thin to ground a gap) but remain available via get_all_papers();
they are counted in ``stats["skipped_no_abstract"]``.

Collected papers carry run-local IDs.  If two *different* papers (not
linked by any signature) share a paper_id, the later one is renamed to
``<paper_id>~<n>`` so chunk IDs and lookups stay unambiguous; the original
is kept in ``metadata["original_paper_id"]`` and counted in
``stats["paper_id_collisions"]``.
"""

from __future__ import annotations

import re
from typing import Optional

from app.core.logging import get_logger
from app.domain.models.collection import CollectedPaper, CollectionResult
from app.domain.models.paper import DocumentChunk, Paper
from app.ingestion.deduplication.normalization import (
    normalize_doi,
    normalize_title_authors_year,
)

logger = get_logger(__name__)

ABSTRACT_SECTION = "Abstract"
CHUNK_TYPE = "title_abstract"
_WHITESPACE_RE = re.compile(r"\s+")


def _clean_text(text: Optional[str]) -> str:
    return _WHITESPACE_RE.sub(" ", text or "").strip()


def _signatures(cp: CollectedPaper) -> list[str]:
    """Deduplication signatures for one record (same kinds as the DB deduplicator)."""
    paper = cp.paper
    signatures: list[str] = []
    doi = normalize_doi(paper.doi)
    if doi:
        signatures.append(f"doi:{doi}")
    tay = normalize_title_authors_year(
        paper.title,
        [a.model_dump() for a in paper.authors],
        paper.publication_year,
    )
    if tay:
        signatures.append(f"tay:{tay}")
    if cp.provenance.source and cp.provenance.provider_id:
        signatures.append(f"prov:{cp.provenance.source}:{cp.provenance.provider_id}")
    return signatures


def _group_duplicates(collected: list[CollectedPaper]) -> list[list[CollectedPaper]]:
    """
    Merge records connected through shared signatures (union-find).

    Groups are ordered by their first record, and records within a group
    keep their collection order, so the result is deterministic.
    """
    parent = list(range(len(collected)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    first_with_signature: dict[str, int] = {}
    for index, cp in enumerate(collected):
        for signature in _signatures(cp):
            other = first_with_signature.setdefault(signature, index)
            root_a, root_b = find(index), find(other)
            if root_a != root_b:
                parent[max(root_a, root_b)] = min(root_a, root_b)

    groups: dict[int, list[CollectedPaper]] = {}
    for index, cp in enumerate(collected):
        groups.setdefault(find(index), []).append(cp)
    return list(groups.values())


def _merge_group(group: list[CollectedPaper]) -> Paper:
    """Representative = longest abstract; fill missing fields from the group."""
    representative = max(group, key=lambda cp: len(_clean_text(cp.paper.abstract))).paper
    updates: dict = {}
    for field_name in ("doi", "publication_year", "paper_url", "venue"):
        if not getattr(representative, field_name):
            value = next(
                (getattr(cp.paper, field_name) for cp in group if getattr(cp.paper, field_name)),
                None,
            )
            if value:
                updates[field_name] = value
    if not representative.authors:
        authors = next((cp.paper.authors for cp in group if cp.paper.authors), None)
        if authors:
            updates["authors"] = authors
    if not _clean_text(representative.title):
        title = next((cp.paper.title for cp in group if _clean_text(cp.paper.title)), None)
        if title:
            updates["title"] = title
    return representative.model_copy(update=updates) if updates else representative


class CollectionCorpusStore:
    """
    In-memory CorpusStore over the unique papers of a CollectionResult.

    Build with ``CollectionCorpusStore.from_collection_result(result)``.
    Chunk IDs are ``"<paper_id>:abstract"`` and every chunk carries its
    provenance in ``metadata`` (title, DOI, year, providers, provider IDs,
    source queries, paper URL, merged duplicate count).
    """

    def __init__(
        self,
        papers: list[Paper],
        chunks: list[DocumentChunk],
        stats: dict,
    ) -> None:
        self._papers = list(papers)
        self._chunks = list(chunks)
        self._papers_by_id = {p.paper_id: p for p in self._papers}
        self._chunks_by_id = {c.chunk_id: c for c in self._chunks}
        self._chunks_by_paper: dict[str, list[DocumentChunk]] = {}
        for chunk in self._chunks:
            self._chunks_by_paper.setdefault(chunk.paper_id, []).append(chunk)
        self.stats = dict(stats)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    @classmethod
    def from_collection_result(
        cls, result: Optional[CollectionResult]
    ) -> CollectionCorpusStore:
        """Build the corpus; ``None`` or an empty result yields an empty store."""
        collected = list(result.papers) if result is not None else []

        papers: list[Paper] = []
        chunks: list[DocumentChunk] = []
        skipped_no_abstract = 0
        collisions = 0
        used_ids: set[str] = set()
        for group in _group_duplicates(collected):
            paper = _merge_group(group)
            metadata = cls._provenance(paper, group)

            if paper.paper_id in used_ids:
                collisions += 1
                suffix = 2
                while f"{paper.paper_id}~{suffix}" in used_ids:
                    suffix += 1
                new_id = f"{paper.paper_id}~{suffix}"
                logger.warning(
                    "CollectionCorpusStore: paper_id collision between different papers; "
                    "renaming",
                    extra={"paper_id": paper.paper_id, "renamed_to": new_id},
                )
                metadata["original_paper_id"] = paper.paper_id
                paper = paper.model_copy(update={"paper_id": new_id})
            used_ids.add(paper.paper_id)
            papers.append(paper)

            abstract = _clean_text(paper.abstract)
            if not abstract:
                skipped_no_abstract += 1
                continue
            title = _clean_text(paper.title)
            text = f"{title}\n\n{abstract}" if title else abstract
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{paper.paper_id}:abstract",
                    paper_id=paper.paper_id,
                    section=ABSTRACT_SECTION,
                    text=text,
                    metadata=metadata,
                )
            )

        stats = {
            "collected_records": len(collected),
            "unique_papers": len(papers),
            "chunks": len(chunks),
            "skipped_no_abstract": skipped_no_abstract,
            "duplicates_merged": len(collected) - len(papers),
            "paper_id_collisions": collisions,
        }
        logger.info("CollectionCorpusStore: corpus built", extra=stats)
        return cls(papers, chunks, stats)

    @staticmethod
    def _provenance(paper: Paper, group: list[CollectedPaper]) -> dict:
        def unique(values: list) -> list:
            return list(dict.fromkeys(v for v in values if v))

        return {
            "corpus": "collection",
            "chunk_type": CHUNK_TYPE,
            "title": _clean_text(paper.title),
            "doi": normalize_doi(paper.doi),
            "publication_year": paper.publication_year,
            "paper_url": paper.paper_url,
            "providers": unique([cp.provenance.source for cp in group]),
            "provider_ids": unique(
                [f"{cp.provenance.source}:{cp.provenance.provider_id}" for cp in group]
            ),
            "source_queries": unique([cp.provenance.source_query for cp in group]),
            "duplicates_merged": len(group) - 1,
        }

    # ------------------------------------------------------------------ #
    # CorpusStore protocol
    # ------------------------------------------------------------------ #

    def get_all_chunks(self) -> list[DocumentChunk]:
        return list(self._chunks)

    def get_chunks_by_paper_id(self, paper_id: str) -> list[DocumentChunk]:
        return list(self._chunks_by_paper.get(paper_id, []))

    def get_paper_by_id(self, paper_id: str) -> Optional[Paper]:
        return self._papers_by_id.get(paper_id)

    def get_all_papers(self) -> list[Paper]:
        return list(self._papers)

    # ------------------------------------------------------------------ #
    # Extras (same as StubCorpusStore)
    # ------------------------------------------------------------------ #

    def get_chunk_by_id(self, chunk_id: str) -> Optional[DocumentChunk]:
        return self._chunks_by_id.get(chunk_id)
