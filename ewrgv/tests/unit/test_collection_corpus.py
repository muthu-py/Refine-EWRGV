"""
tests/unit/test_collection_corpus.py
--------------------------------------
Unit tests for CollectionCorpusStore (collected titles + abstracts → corpus).

Offline: hand-created CollectionResult fixtures only.
"""

from __future__ import annotations

from app.domain.interfaces import CorpusStore
from app.domain.models.collection import CollectedPaper, CollectionResult, PaperProvenance
from app.domain.models.paper import Author, Paper
from app.ingestion.corpus.collection_corpus import CollectionCorpusStore
from app.retrieval.bm25 import SparseRetriever
from tests.fixtures.sample_collection import (
    CAN_GNN_ABSTRACT,
    CAN_GNN_CHUNK,
    GRID_CHUNK,
    Q_EXPANDED,
    Q_MAIN,
    TRANSFER_ABSTRACT,
    TRANSFER_CHUNK,
    make_collection_result,
)


def _store() -> CollectionCorpusStore:
    return CollectionCorpusStore.from_collection_result(make_collection_result())


def _single(paper: Paper, source: str = "openalex", query: str = "q") -> CollectionResult:
    return CollectionResult(
        query_id="r",
        papers=[CollectedPaper(
            paper=paper,
            provenance=PaperProvenance(source=source, provider_id="X1", source_query=query),
        )],
    )


class TestConstruction:
    def test_satisfies_corpus_store_protocol(self):
        assert isinstance(_store(), CorpusStore)

    def test_stats(self):
        assert _store().stats == {
            "collected_records": 7,
            "unique_papers": 5,
            "chunks": 3,
            "skipped_no_abstract": 2,
            "duplicates_merged": 2,
            "paper_id_collisions": 0,
        }

    def test_one_title_abstract_chunk_per_paper(self):
        chunks = _store().get_all_chunks()
        assert [c.chunk_id for c in chunks] == [CAN_GNN_CHUNK, TRANSFER_CHUNK, GRID_CHUNK]
        can = chunks[0]
        assert can.paper_id == "s2-can-gnn"
        assert can.section == "Abstract"
        assert can.text == (
            "Graph Neural Networks for CAN Bus Intrusion Detection\n\n" + CAN_GNN_ABSTRACT
        )

    def test_chunk_id_format_links_to_paper(self):
        """Chunk IDs are '<paper_id>:abstract' and resolve to their paper.
        (Paper IDs themselves are run-local: collectors assign random UUIDs.)"""
        store = _store()
        for chunk in store.get_all_chunks():
            assert chunk.chunk_id == f"{chunk.paper_id}:abstract"
            assert store.get_paper_by_id(chunk.paper_id) is not None
            assert store.get_chunks_by_paper_id(chunk.paper_id) == [chunk]

    def test_empty_or_none_result(self):
        for result in (None, CollectionResult(query_id="r", status="failed")):
            store = CollectionCorpusStore.from_collection_result(result)
            assert store.get_all_chunks() == []
            assert store.get_all_papers() == []
            assert store.stats["chunks"] == 0
            assert SparseRetriever(corpus_store=store).retrieve("anything") == []


class TestDeduplicationAndProvenance:
    def test_doi_variants_merged_with_longest_abstract(self):
        store = _store()
        assert store.get_paper_by_id("oa-can-gnn") is None  # merged away
        meta = store.get_chunk_by_id(CAN_GNN_CHUNK).metadata
        assert meta["doi"] == "10.1000/can.gnn.2023"
        assert meta["providers"] == ["semantic_scholar", "openalex"]
        assert meta["provider_ids"] == ["semantic_scholar:S2-111", "openalex:W222"]
        assert meta["source_queries"] == [Q_MAIN, Q_EXPANDED]
        assert meta["duplicates_merged"] == 1

    def test_title_authors_year_merge_without_doi(self):
        store = _store()
        assert store.get_paper_by_id("s2-transfer") is None
        chunk = store.get_chunk_by_id(TRANSFER_CHUNK)
        assert chunk.text.endswith(TRANSFER_ABSTRACT)  # non-empty duplicate chosen
        assert chunk.metadata["providers"] == ["openalex", "semantic_scholar"]
        assert chunk.metadata["doi"] is None

    def test_same_title_different_year_not_merged(self):
        def paper(pid: str, year: int) -> CollectedPaper:
            return CollectedPaper(
                paper=Paper(paper_id=pid, title="Same Title", abstract=f"abstract {pid}",
                            publication_year=year),
                provenance=PaperProvenance(source="openalex", provider_id=pid, source_query="q"),
            )
        store = CollectionCorpusStore.from_collection_result(
            CollectionResult(query_id="r", papers=[paper("a", 2020), paper("b", 2021)])
        )
        assert len(store.get_all_chunks()) == 2

    def test_provenance_fields(self):
        meta = _store().get_chunk_by_id(GRID_CHUNK).metadata
        assert meta["corpus"] == "collection"
        assert meta["chunk_type"] == "title_abstract"
        assert meta["title"] == "Federated Learning for Smart Grid Forecasting"
        assert meta["publication_year"] == 2023
        assert meta["paper_url"] == "https://example.org/s2-grid"


class TestMissingData:
    def test_missing_and_blank_abstracts_skipped_but_paper_kept(self):
        store = _store()
        for pid in ("s2-survey", "s2-blank"):
            assert store.get_paper_by_id(pid) is not None
            assert store.get_chunks_by_paper_id(pid) == []
            assert store.get_chunk_by_id(f"{pid}:abstract") is None

    def test_whitespace_normalised(self):
        store = CollectionCorpusStore.from_collection_result(_single(
            Paper(paper_id="p", title="  A   Title ", abstract="Line one.\n\n  Line   two.")
        ))
        assert store.get_all_chunks()[0].text == "A Title\n\nLine one. Line two."

    def test_empty_title_uses_abstract_only(self):
        store = CollectionCorpusStore.from_collection_result(_single(
            Paper(paper_id="p", title="", abstract="Only an abstract.")
        ))
        assert store.get_all_chunks()[0].text == "Only an abstract."

    def test_lookups_for_unknown_ids(self):
        store = _store()
        assert store.get_paper_by_id("nope") is None
        assert store.get_chunk_by_id("nope") is None
        assert store.get_chunks_by_paper_id("nope") == []


class TestBM25OverCollectionCorpus:
    def test_relevant_abstract_ranked_first(self):
        retriever = SparseRetriever(corpus_store=_store())
        assert [c.chunk_id for c in retriever.retrieve(Q_MAIN, top_k=5)] == [
            CAN_GNN_CHUNK, TRANSFER_CHUNK,
        ]  # off-topic grid paper has no matching terms

    def test_expanded_query_ranking(self):
        retriever = SparseRetriever(corpus_store=_store())
        assert retriever.retrieve(Q_EXPANDED, top_k=1)[0].chunk_id == TRANSFER_CHUNK


# ======================================================================
# H1 regression - signature-based, transitive deduplication
# ======================================================================


def _rec(pid, title="Graph Neural Networks for CAN Bus Intrusion Detection", abstract="abstract",
         source="openalex", provider_id=None, doi=None, year=2023,
         authors=("Arun Kumar",), query="q", url=None) -> CollectedPaper:
    return CollectedPaper(
        paper=Paper(paper_id=pid, title=title, abstract=abstract, doi=doi,
                    publication_year=year, authors=[Author(name=a) for a in authors],
                    paper_url=url),
        provenance=PaperProvenance(source=source, provider_id=provider_id or f"P-{pid}",
                                   source_query=query),
    )


def _build(*records: CollectedPaper) -> CollectionCorpusStore:
    return CollectionCorpusStore.from_collection_result(
        CollectionResult(query_id="r", papers=list(records))
    )


class TestSignatureDeduplication:
    def test_doi_present_on_only_one_record(self):
        """Linked by title+authors+year even though only one record has a DOI."""
        store = _build(
            _rec("s2", abstract="short", source="semantic_scholar", doi=None),
            _rec("oa", abstract="the much longer abstract", source="openalex", doi="10.1/X"),
        )
        chunks = store.get_all_chunks()
        assert [c.chunk_id for c in chunks] == ["oa:abstract"]
        assert chunks[0].metadata["doi"] == "10.1/x"
        assert chunks[0].metadata["providers"] == ["semantic_scholar", "openalex"]

    def test_doi_filled_from_duplicate_when_representative_lacks_it(self):
        store = _build(
            _rec("s2", abstract="the much longer abstract", source="semantic_scholar", doi=None,
                 url="https://s2/x"),
            _rec("oa", abstract="short", source="openalex", doi="https://doi.org/10.1/X"),
        )
        [chunk] = store.get_all_chunks()
        assert chunk.paper_id == "s2"  # longest abstract wins
        assert chunk.metadata["doi"] == "10.1/x"  # filled from the OpenAlex record
        assert store.get_paper_by_id("s2").doi == "https://doi.org/10.1/X"
        assert chunk.metadata["paper_url"] == "https://s2/x"

    def test_year_filled_from_duplicate(self):
        store = _build(
            _rec("a", abstract="longer abstract here", year=None, provider_id="S2-1",
                 source="semantic_scholar"),
            _rec("b", abstract="short", year=2021, provider_id="S2-1", source="semantic_scholar"),
        )
        [chunk] = store.get_all_chunks()
        assert chunk.metadata["publication_year"] == 2021

    def test_same_provider_record_without_year_or_doi(self):
        """Same provider record from two expanded queries: linked by provider ID."""
        store = _build(
            _rec("u1", source="semantic_scholar", provider_id="S2-SAME", year=None, query="q1"),
            _rec("u2", source="semantic_scholar", provider_id="S2-SAME", year=None, query="q2"),
        )
        [chunk] = store.get_all_chunks()
        assert chunk.metadata["source_queries"] == ["q1", "q2"]
        assert chunk.metadata["duplicates_merged"] == 1

    def test_transitive_matching(self):
        """A-B share a DOI, B-C share a provider ID, C-D share title+authors+year."""
        store = _build(
            _rec("a", source="openalex", provider_id="W1", doi="10.1/t", year=None),
            _rec("b", source="semantic_scholar", provider_id="S2-9", doi="10.1/T", year=None,
                 title="Different Title Variant"),
            _rec("c", source="semantic_scholar", provider_id="S2-9", doi=None, year=2024,
                 title="Third Title"),
            _rec("d", source="crossref", provider_id="C-4", doi=None, year=2024,
                 title="Third Title", abstract="the longest abstract of the group"),
        )
        assert store.stats["unique_papers"] == 1
        [chunk] = store.get_all_chunks()
        assert chunk.paper_id == "d"
        assert chunk.metadata["providers"] == ["openalex", "semantic_scholar", "crossref"]
        assert chunk.metadata["duplicates_merged"] == 3
        assert chunk.metadata["doi"] == "10.1/t"

    def test_unrelated_papers_not_merged(self):
        store = _build(
            _rec("a", title="Paper One", doi="10.1/a", provider_id="W1"),
            _rec("b", title="Paper Two", doi="10.1/b", provider_id="W2"),
            _rec("c", title="Paper One", doi=None, year=None, provider_id="W3"),  # no shared signature
        )
        assert store.stats["unique_papers"] == 3

    def test_known_limitation_author_format_differences_not_merged(self):
        """Documented limitation shared with the database deduplicator."""
        store = _build(
            _rec("a", authors=("A. Kumar",), source="semantic_scholar", provider_id="S2-1"),
            _rec("b", authors=("Arun Kumar",), source="openalex", provider_id="W1"),
        )
        assert store.stats["unique_papers"] == 2


# ======================================================================
# L1 regression - paper_id collisions
# ======================================================================


class TestPaperIdCollisions:
    def test_different_papers_sharing_id_are_disambiguated(self):
        store = _build(
            _rec("same", title="Paper One", abstract="abstract one", doi="10.1/a", provider_id="W1"),
            _rec("same", title="Paper Two", abstract="abstract two", doi="10.1/b", provider_id="W2"),
        )
        chunks = store.get_all_chunks()
        assert [c.chunk_id for c in chunks] == ["same:abstract", "same~2:abstract"]
        assert store.get_chunk_by_id("same:abstract").text.startswith("Paper One")
        assert store.get_chunk_by_id("same~2:abstract").text.startswith("Paper Two")
        assert store.get_paper_by_id("same").title == "Paper One"
        assert store.get_paper_by_id("same~2").title == "Paper Two"
        assert chunks[1].metadata["original_paper_id"] == "same"
        assert store.stats["paper_id_collisions"] == 1

    def test_renamed_id_does_not_clash_with_existing_id(self):
        store = _build(
            _rec("x", title="One", doi="10.1/1", provider_id="W1"),
            _rec("x~2", title="Two", doi="10.1/2", provider_id="W2"),
            _rec("x", title="Three", doi="10.1/3", provider_id="W3"),
        )
        assert sorted(p.paper_id for p in store.get_all_papers()) == ["x", "x~2", "x~3"]

    def test_duplicates_sharing_id_are_not_a_collision(self):
        store = _build(
            _rec("same", doi="10.1/a", provider_id="W1", source="openalex"),
            _rec("same", doi="10.1/a", provider_id="S1", source="semantic_scholar"),
        )
        assert store.stats["paper_id_collisions"] == 0
        assert len(store.get_all_chunks()) == 1


# ======================================================================
# M1 regression - no database driver needed
# ======================================================================


class TestNoDatabaseDependency:
    def test_imports_without_asyncpg(self):
        """
        ResearchPipeline and CollectionCorpusStore import with asyncpg absent.

        The subprocess runs from the project directory (resolved from this
        file), with it on PYTHONPATH, so the result does not depend on the
        directory pytest was launched from.
        """
        import os
        import subprocess
        import sys
        from pathlib import Path

        project_dir = Path(__file__).resolve().parents[2]  # .../ewrgv
        assert (project_dir / "app" / "orchestration" / "research_pipeline.py").is_file()

        code = (
            "import sys; sys.modules['asyncpg'] = None\n"
            "from app.ingestion.corpus.collection_corpus import CollectionCorpusStore\n"
            "from app.orchestration.research_pipeline import ResearchPipeline\n"
            "print('ok')\n"
        )
        env = {**os.environ, "PYTHONPATH": str(project_dir)}
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=project_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert "asyncpg" not in proc.stderr, (
            "importing the pipeline/corpus requires asyncpg:\n" + proc.stderr
        )
        assert proc.returncode == 0, "import failed (not asyncpg-related):\n" + proc.stderr
        assert proc.stdout.strip().endswith("ok")

    def test_service_reexports_same_helpers(self):
        from app.ingestion.deduplication import normalization, service

        assert service.normalize_doi is normalization.normalize_doi
        assert service.normalize_title_authors_year is normalization.normalize_title_authors_year
