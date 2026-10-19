"""
app/ingestion/corpus/run_corpus.py
------------------------------------
RunCorpus — the retrieval corpus of one research run, built from the
artifacts produced by Phases 2.7 (chunking) and 2.9 (representation).

* Chunks are the canonical PaperChunk objects from each paper's
  ``chunks.json`` (full-text, section-aware, with page provenance).
* Papers carry the bibliographic fields (title, year, DOI, authors) that
  PaperChunk does not, so retrieval and gap detection can look them up.
* Dense embeddings come from the ``chunk_embeddings`` table and the BM25
  index from the file persisted by RepresentationService.  Both are
  optional: a missing BM25 index is rebuilt from the chunks, and chunks
  without a stored embedding are embedded on demand by DenseRetriever.

Satisfies the CorpusStore protocol (app.domain.interfaces).

RunCorpusLoader builds a RunCorpus asynchronously from Supabase Storage and
PostgreSQL.  RunCorpus itself does no I/O, so tests can construct it
directly from in-memory chunks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.domain.models.paper import Author, Paper
from app.domain.models.paper_chunk import ChunkingResult, PaperChunk
from app.retrieval.bm25_index import BM25Index
from app.storage.artifact_paths import ARTIFACT_BUCKET, get_chunks_path

logger = get_logger(__name__)


def bm25_index_path(run_id: str) -> Path:
    """Location of a run's persisted BM25 index (written by RepresentationService)."""
    return Path(settings.BM25_INDEX_PATH) / run_id / "bm25_index.pkl"


class RunCorpus:
    """
    In-memory CorpusStore over the PaperChunks of one research run.

    Parameters
    ----------
    papers:
        Papers of the run (bibliographic metadata).
    chunks:
        PaperChunks of those papers.  Duplicate chunk IDs keep the first.
    embeddings:
        Optional ``chunk_id → vector`` map of stored dense embeddings.
    bm25_index:
        Optional prebuilt BM25Index over the same chunks.  Built lazily
        from ``chunks`` when not supplied.
    run_id:
        The research run this corpus belongs to.
    """

    def __init__(
        self,
        papers: list[Paper],
        chunks: list[PaperChunk],
        embeddings: Optional[dict[str, list[float]]] = None,
        bm25_index: Optional[BM25Index] = None,
        run_id: Optional[str] = None,
    ) -> None:
        self.run_id = run_id
        self._papers = list(papers)
        self._papers_by_id = {p.paper_id: p for p in self._papers}
        self._chunks: list[PaperChunk] = []
        self._chunks_by_id: dict[str, PaperChunk] = {}
        self._chunks_by_paper: dict[str, list[PaperChunk]] = {}
        for chunk in chunks:
            if chunk.chunk_id in self._chunks_by_id:
                continue
            self._chunks.append(chunk)
            self._chunks_by_id[chunk.chunk_id] = chunk
            self._chunks_by_paper.setdefault(chunk.paper_id, []).append(chunk)
        self._embeddings = {
            cid: vec for cid, vec in (embeddings or {}).items() if cid in self._chunks_by_id
        }
        self._bm25_index = bm25_index

    # ------------------------------------------------------------------ #
    # CorpusStore protocol
    # ------------------------------------------------------------------ #

    def get_all_chunks(self) -> list[PaperChunk]:
        return list(self._chunks)

    def get_chunks_by_paper_id(self, paper_id: str) -> list[PaperChunk]:
        return list(self._chunks_by_paper.get(paper_id, []))

    def get_paper_by_id(self, paper_id: str) -> Optional[Paper]:
        return self._papers_by_id.get(paper_id)

    def get_all_papers(self) -> list[Paper]:
        return list(self._papers)

    def get_chunk_by_id(self, chunk_id: str) -> Optional[PaperChunk]:
        return self._chunks_by_id.get(chunk_id)

    # ------------------------------------------------------------------ #
    # Representations
    # ------------------------------------------------------------------ #

    def get_embedding(self, chunk_id: str) -> Optional[list[float]]:
        """Stored dense embedding for a chunk, or None."""
        return self._embeddings.get(chunk_id)

    @property
    def bm25_index(self) -> BM25Index:
        """The run's BM25 index; built from the chunks if none was loaded."""
        if self._bm25_index is None:
            self._bm25_index = BM25Index().build(self._chunks)
        return self._bm25_index

    @property
    def stats(self) -> dict:
        section_types: dict[str, int] = {}
        for chunk in self._chunks:
            key = chunk.section_type.value
            section_types[key] = section_types.get(key, 0) + 1
        return {
            "papers": len(self._papers),
            "papers_with_chunks": len(self._chunks_by_paper),
            "chunks": len(self._chunks),
            "embedded_chunks": len(self._embeddings),
            "chunks_by_section_type": section_types,
        }


def paper_from_row(row: dict) -> Paper:
    """Map a ``papers`` table row to the Paper domain model."""
    authors_raw: Any = row.get("authors") or []
    if isinstance(authors_raw, str):
        try:
            authors_raw = json.loads(authors_raw)
        except json.JSONDecodeError:
            authors_raw = []
    authors = [
        Author(**a) if isinstance(a, dict) else Author(name=str(a))
        for a in authors_raw
        if a
    ]
    return Paper(
        paper_id=str(row["paper_id"]),
        title=row.get("title") or "",
        abstract=row.get("abstract") or "",
        doi=row.get("doi"),
        publication_year=row.get("publication_year"),
        authors=authors,
    )


class RunCorpusLoader:
    """
    Loads a RunCorpus for a research run from storage.

    Parameters
    ----------
    paper_repo:
        PaperRepository (``get_papers_for_corpus``).
    storage_client:
        SupabaseStorageClient used to download ``chunks.json``.
    embedding_repo:
        Optional EmbeddingRepository; when given, stored embeddings for
        ``embedding_model`` are loaded for dense retrieval.
    embedding_model:
        Model whose embeddings to load (defaults to settings.EMBEDDING_MODEL).
    """

    def __init__(
        self,
        paper_repo: Any,
        storage_client: Any,
        embedding_repo: Any = None,
        embedding_model: Optional[str] = None,
    ) -> None:
        self._paper_repo = paper_repo
        self._storage = storage_client
        self._embedding_repo = embedding_repo
        self._embedding_model = embedding_model or settings.EMBEDDING_MODEL

    async def load(self, run_id: str) -> RunCorpus:
        rows = await self._paper_repo.get_papers_for_corpus(run_id)
        papers: list[Paper] = []
        chunks: list[PaperChunk] = []
        failed = 0
        for row in rows:
            paper = paper_from_row(row)
            papers.append(paper)
            path = row.get("chunks_storage_path") or get_chunks_path(paper.paper_id)
            try:
                raw = await self._storage.download(ARTIFACT_BUCKET, path)
                result = ChunkingResult.model_validate(json.loads(raw))
            except Exception as exc:  # noqa: BLE001 — one bad artifact must not sink the run
                failed += 1
                logger.warning(
                    "RunCorpusLoader: could not load chunks for paper %s: %s",
                    paper.paper_id,
                    exc,
                )
                continue
            chunks.extend(result.chunks)

        embeddings: dict[str, list[float]] = {}
        if self._embedding_repo is not None and papers:
            embeddings = await self._embedding_repo.get_embeddings_for_papers(
                [p.paper_id for p in papers], self._embedding_model
            )

        bm25_index: Optional[BM25Index] = None
        index_path = bm25_index_path(run_id)
        if index_path.exists():
            try:
                bm25_index = BM25Index.load(index_path)
            except Exception as exc:  # noqa: BLE001 — fall back to rebuilding
                logger.warning("RunCorpusLoader: BM25 index unreadable, rebuilding: %s", exc)
        corpus = RunCorpus(papers, chunks, embeddings, bm25_index=bm25_index, run_id=run_id)
        if bm25_index is not None and set(bm25_index.chunk_ids) != {
            c.chunk_id for c in corpus.get_all_chunks()
        }:
            # Chunks were regenerated after the index was built; IDs no longer match.
            logger.warning("RunCorpusLoader: BM25 index is stale for run %s; rebuilding", run_id)
            corpus = RunCorpus(papers, chunks, embeddings, run_id=run_id)

        logger.info(
            "RunCorpusLoader: corpus loaded",
            extra={"run_id": run_id, "failed_papers": failed, **corpus.stats},
        )
        return corpus
