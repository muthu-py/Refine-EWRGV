"""
tests/unit/test_run_corpus.py
-------------------------------
Unit tests for RunCorpus and RunCorpusLoader (app/ingestion/corpus/run_corpus.py).

The loader is exercised with mocked PaperRepository, storage client and
EmbeddingRepository, and a temporary BM25_INDEX_PATH.  No database or
network.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import settings
from app.domain.enums import SectionType
from app.domain.models.paper_chunk import ChunkingResult
from app.ingestion.corpus.run_corpus import (
    RunCorpus,
    RunCorpusLoader,
    bm25_index_path,
    paper_from_row,
)
from app.retrieval.bm25_index import BM25Index
from tests.fixtures.sample_corpus import make_sample_corpus, sample_chunks, sample_papers

RUN = "run-42"


# ======================================================================
# RunCorpus
# ======================================================================


class TestRunCorpus:
    def test_corpus_store_protocol(self):
        corpus = make_sample_corpus()
        assert len(corpus.get_all_chunks()) == 13
        assert len(corpus.get_all_papers()) == 5
        assert corpus.get_paper_by_id("paper-003").publication_year == 2020
        assert {c.paper_id for c in corpus.get_chunks_by_paper_id("paper-003")} == {"paper-003"}
        assert corpus.get_chunk_by_id("chunk-003-limits").section_type == SectionType.LIMITATIONS
        assert corpus.get_chunk_by_id("missing") is None
        assert corpus.get_paper_by_id("missing") is None

    def test_duplicate_chunk_ids_keep_first(self):
        chunks = sample_chunks()
        dup = chunks[0].model_copy(update={"text": "different"})
        corpus = RunCorpus(sample_papers(), chunks + [dup])
        assert len(corpus.get_all_chunks()) == 13
        assert corpus.get_chunk_by_id(chunks[0].chunk_id).text == chunks[0].text

    def test_embeddings_filtered_to_corpus_chunks(self):
        corpus = make_sample_corpus(embeddings={"chunk-003-limits": [1.0], "ghost": [0.0]})
        assert corpus.get_embedding("chunk-003-limits") == [1.0]
        assert corpus.get_embedding("ghost") is None
        assert corpus.stats["embedded_chunks"] == 1

    def test_bm25_index_built_lazily(self):
        corpus = make_sample_corpus()
        index = corpus.bm25_index
        assert isinstance(index, BM25Index)
        assert set(index.chunk_ids) == {c.chunk_id for c in corpus.get_all_chunks()}
        assert corpus.bm25_index is index

    def test_stats(self):
        stats = make_sample_corpus().stats
        assert stats["papers"] == 5
        assert stats["papers_with_chunks"] == 5
        assert stats["chunks"] == 13
        assert stats["chunks_by_section_type"]["LIMITATIONS"] == 1

    def test_paper_from_row_parses_authors_json(self):
        paper = paper_from_row({
            "paper_id": 7,
            "title": "T",
            "abstract": None,
            "doi": "10.1/x",
            "publication_year": 2021,
            "authors": json.dumps([{"name": "A"}, "B", None]),
        })
        assert paper.paper_id == "7"
        assert paper.abstract == ""
        assert [a.name for a in paper.authors] == ["A", "B"]

    def test_paper_from_row_bad_authors_json(self):
        assert paper_from_row({"paper_id": "p", "authors": "{not json"}).authors == []


# ======================================================================
# RunCorpusLoader
# ======================================================================


def _rows() -> list[dict]:
    return [
        {
            "paper_id": p.paper_id,
            "title": p.title,
            "abstract": p.abstract,
            "doi": p.doi,
            "publication_year": p.publication_year,
            "authors": [],
            "chunks_storage_path": f"papers/{p.paper_id}/chunks.json",
        }
        for p in sample_papers()
    ]


def _chunks_json(paper_id: str) -> bytes:
    result = ChunkingResult(
        paper_id=paper_id,
        source_sections_path=f"papers/{paper_id}/sections.json",
        chunks=[c for c in sample_chunks() if c.paper_id == paper_id],
    )
    return result.model_dump_json().encode()


@pytest.fixture
def bm25_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "BM25_INDEX_PATH", str(tmp_path))
    return tmp_path


def _loader(rows=None, download=None, embedding_repo=None):
    paper_repo = MagicMock()
    paper_repo.get_papers_for_corpus = AsyncMock(return_value=_rows() if rows is None else rows)
    storage = MagicMock()

    async def default_download(bucket, path):
        return _chunks_json(path.split("/")[1])

    storage.download = AsyncMock(side_effect=download or default_download)
    loader = RunCorpusLoader(paper_repo, storage, embedding_repo, embedding_model="m")
    return loader, paper_repo, storage


class TestRunCorpusLoader:
    async def test_loads_chunks_and_papers(self, bm25_dir):
        loader, paper_repo, storage = _loader()
        corpus = await loader.load(RUN)
        paper_repo.get_papers_for_corpus.assert_awaited_once_with(RUN)
        assert storage.download.await_count == 5
        assert corpus.run_id == RUN
        assert corpus.stats["chunks"] == 13
        assert corpus.get_paper_by_id("paper-003").publication_year == 2020

    async def test_default_chunks_path_when_column_empty(self, bm25_dir):
        rows = _rows()[:1]
        rows[0]["chunks_storage_path"] = None
        loader, _, storage = _loader(rows=rows)
        await loader.load(RUN)
        assert storage.download.await_args.args[1] == "papers/paper-001/chunks.json"

    async def test_failed_paper_skipped(self, bm25_dir):
        async def download(bucket, path):
            if "paper-002" in path:
                raise OSError("404")
            return _chunks_json(path.split("/")[1])

        loader, _, _ = _loader(download=download)
        corpus = await loader.load(RUN)
        assert corpus.get_chunks_by_paper_id("paper-002") == []
        assert corpus.get_paper_by_id("paper-002") is not None  # paper still listed
        assert corpus.stats["papers_with_chunks"] == 4

    async def test_embeddings_loaded_for_model(self, bm25_dir):
        embedding_repo = MagicMock()
        embedding_repo.get_embeddings_for_papers = AsyncMock(
            return_value={"chunk-003-limits": [0.5, 0.5]}
        )
        loader, _, _ = _loader(embedding_repo=embedding_repo)
        corpus = await loader.load(RUN)
        ids, model = embedding_repo.get_embeddings_for_papers.await_args.args
        assert sorted(ids) == [f"paper-00{i}" for i in range(1, 6)]
        assert model == "m"
        assert corpus.get_embedding("chunk-003-limits") == [0.5, 0.5]

    async def test_persisted_bm25_index_used(self, bm25_dir):
        index = BM25Index().build(sample_chunks())
        path = bm25_index_path(RUN)
        path.parent.mkdir(parents=True)
        index.save(path)
        loader, _, _ = _loader()
        corpus = await loader.load(RUN)
        assert corpus._bm25_index is not None  # loaded, not lazily rebuilt
        assert set(corpus.bm25_index.chunk_ids) == {c.chunk_id for c in sample_chunks()}

    async def test_stale_bm25_index_rebuilt(self, bm25_dir):
        index = BM25Index().build(sample_chunks()[:3])  # chunks regenerated since
        path = bm25_index_path(RUN)
        path.parent.mkdir(parents=True)
        index.save(path)
        loader, _, _ = _loader()
        corpus = await loader.load(RUN)
        assert corpus._bm25_index is None
        assert len(corpus.bm25_index.chunk_ids) == 13

    async def test_unreadable_bm25_index_rebuilt(self, bm25_dir):
        path = bm25_index_path(RUN)
        path.parent.mkdir(parents=True)
        path.write_bytes(b"not a pickle")
        loader, _, _ = _loader()
        corpus = await loader.load(RUN)
        assert len(corpus.bm25_index.chunk_ids) == 13

    async def test_empty_run(self, bm25_dir):
        loader, _, storage = _loader(rows=[])
        corpus = await loader.load(RUN)
        assert corpus.get_all_chunks() == []
        storage.download.assert_not_awaited()
