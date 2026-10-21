"""
tests/unit/test_retrieval_fixes.py
------------------------------------
Retrieval-stage fixes:

* A1 /query is stored; /collect {query_id} searches the question + ALL
  expanded queries, honours the limits, reports ``capped``, fails loudly
  when the run cannot be saved.
* A2 references / acknowledgments / boilerplate are excluded by default.
* A3 section_types (incl. the gap_evidence preset), paper and year filters.
* A4 the run corpus is cached and cleared by the chunk / represent workers.
* A5 multi-query retrieval fuses the question + expansions with RRF.
"""

from __future__ import annotations

import asyncio
from collections import Counter
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import app.api.routes.research as research_routes
from app.core.config import settings
from app.domain.enums import SectionType
from app.domain.models.collection import CollectionResult
from app.domain.models.paper_chunk import PaperChunk
from app.ingestion.collectors.service import LiteratureCollectionService
from app.ingestion.corpus.run_corpus import RunCorpus
from app.ingestion.sectioning.heading_detector import detect_headings
from app.main import app
from app.retrieval import corpus_cache
from app.retrieval.filters import (
    NO_FILTERS,
    RetrievalFilters,
    expand_section_types,
    is_noise,
)
from app.retrieval.service import RetrievalService
from tests.fixtures.sample_corpus import FakeEmbeddingProvider, sample_chunks, sample_papers

REFERENCE_LIST = (
    "22. Chen Q, Leng T, Zheng L et al. Automated drusen segmentation in SD-OCT images. "
    "Med Image Anal. 2018;46:1-12. doi: 10.1016/j.media.2018.02.001. 23. Wang Y, Zhang Y. "
    "Retinal vessel segmentation. IEEE Trans Med Imaging. 2019;38(4):1-10. 24. Liu Z, Mao H, "
    "Wu CY et al. A convnet for the 2020s. In: Proceedings of the IEEE/CVF CVPR (2022) pp. 11976-86. "
    "language model attention transformer"
)


def _chunk(cid: str, paper: str, stype: SectionType, text: str) -> PaperChunk:
    return PaperChunk(
        chunk_id=cid, paper_id=paper, section_id=f"sec-{cid}", section_title=stype.value.title(),
        section_type=stype, text=text,
    )


def _corpus(extra: list[PaperChunk] | None = None) -> RunCorpus:
    return RunCorpus(sample_papers(), sample_chunks() + (extra or []), run_id="run-x")


# ------------------------------------------------------------------ #
# A2 noise
# ------------------------------------------------------------------ #


class TestNoise:
    def test_reference_sections_are_noise(self) -> None:
        assert is_noise("REFERENCES", "Anything at all")
        assert is_noise("ACKNOWLEDGMENTS", "We thank the reviewers")

    def test_mislabelled_reference_list_is_noise(self) -> None:
        assert is_noise("METHODS", REFERENCE_LIST)

    def test_page_header_is_noise(self) -> None:
        assert is_noise("OTHER", "Volume 5, Issue 12, December 2025 Copyright to IJETIR DOI: 10.48175/IJETIR-9239")

    def test_conclusion_with_licence_banner_is_kept(self) -> None:
        text = ("We show that combining deep learning models with ultra-widefield images achieves "
                "competitive accuracy for early retinal disease detection, although external validation "
                "on multi-centre cohorts and other camera types remains necessary before clinical use. ") * 3
        text += "CC-BY 4.0 International license. medRxiv preprint. The copyright holder for this preprint."
        assert not is_noise("CONCLUSION", text)

    def test_prose_with_a_few_citations_is_kept(self) -> None:
        text = ("Prior work by Smith et al. (2020) and Lee et al. (2021) reported strong accuracy, but "
                "both evaluated on a single public dataset with no external test set, so generalisation "
                "to new populations and imaging devices remains an open question for the field. ") * 2
        assert not is_noise("DISCUSSION", text)


# ------------------------------------------------------------------ #
# A3 filters
# ------------------------------------------------------------------ #


class TestFilters:
    def test_gap_evidence_preset_expands(self) -> None:
        assert expand_section_types(["gap_evidence"]) == frozenset(
            {"LIMITATIONS", "FUTURE_WORK", "DISCUSSION", "CONCLUSION", "RESULTS"}
        )
        assert expand_section_types(["limitations", "gap_evidence"]) >= {"LIMITATIONS", "RESULTS"}
        assert expand_section_types([]) is None

    def test_default_excludes_noise_and_no_filters_keeps_it(self) -> None:
        noise = _chunk("ref-1", "paper-001", SectionType.REFERENCES, REFERENCE_LIST)
        service = RetrievalService(_corpus([noise]), FakeEmbeddingProvider())
        default = service.retrieve("language model attention transformer", top_k=20, method="sparse")
        everything = service.retrieve("language model attention transformer", top_k=20, method="sparse",
                                      filters=NO_FILTERS)
        assert "ref-1" not in {r.chunk_id for r in default}
        assert "ref-1" in {r.chunk_id for r in everything}

    @pytest.mark.parametrize("method", ["sparse", "dense", "hybrid"])
    def test_section_filter_only_returns_allowed_types(self, method: str) -> None:
        service = RetrievalService(_corpus(), FakeEmbeddingProvider())
        flt = RetrievalFilters.build(section_types=["gap_evidence"])
        results = service.retrieve("limitations of language models", top_k=10, method=method, filters=flt)
        assert results, "the sample corpus has RESULTS / LIMITATIONS chunks"
        assert {r.section_type for r in results} <= flt.section_types

    def test_paper_and_year_filters(self) -> None:
        service = RetrievalService(_corpus(), FakeEmbeddingProvider())
        only = service.retrieve("language model", top_k=10, method="sparse",
                                filters=RetrievalFilters.build(paper_ids=["paper-001"]))
        assert only and {r.paper_id for r in only} == {"paper-001"}
        years = [r.metadata.get("publication_year") for r in service.retrieve("language model", top_k=10, method="sparse")]
        cutoff = max(y for y in years if isinstance(y, int))
        recent = service.retrieve("language model", top_k=10, method="sparse",
                                  filters=RetrievalFilters.build(year_from=cutoff))
        assert recent and all(r.metadata["publication_year"] >= cutoff for r in recent)


# ------------------------------------------------------------------ #
# A5 multi-query
# ------------------------------------------------------------------ #


class TestMultiQuery:
    def test_fuses_queries_and_records_matches(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "RETRIEVAL_MAX_PER_PAPER", 2)
        service = RetrievalService(_corpus(), FakeEmbeddingProvider())
        queries = ["transformer attention", "language model limitations", "transformer attention"]
        results = asyncio.run(service.aretrieve_many(queries, top_k=6, method="hybrid"))
        assert results and len(results) <= 6
        assert [r.rank for r in results] == list(range(1, len(results) + 1))
        assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)
        assert all(r.retrieval_method == "multi_hybrid" for r in results)
        matched = {q for r in results for q in r.metadata["matched_queries"]}
        assert matched <= {"transformer attention", "language model limitations"}
        assert len(matched) == 2
        papers = Counter(r.paper_id for r in results)
        if len(papers) * 2 >= len(results):
            assert max(papers.values()) <= 2

    def test_chunk_matched_by_more_queries_ranks_higher(self) -> None:
        service = RetrievalService(_corpus(), FakeEmbeddingProvider())
        single = asyncio.run(service.aretrieve_many(["transformer attention"], top_k=3, method="sparse"))
        top = single[0].chunk_id
        multi = asyncio.run(service.aretrieve_many(
            ["transformer attention", "attention transformer model"], top_k=3, method="sparse"))
        assert multi[0].chunk_id == top
        assert len(multi[0].metadata["matched_queries"]) == 2

    def test_requires_a_query(self) -> None:
        from app.core.exceptions import BadRequestError

        service = RetrievalService(_corpus(), FakeEmbeddingProvider())
        with pytest.raises(BadRequestError):
            asyncio.run(service.aretrieve_many(["  "], top_k=3, method="sparse"))


# ------------------------------------------------------------------ #
# A4 corpus cache
# ------------------------------------------------------------------ #


class TestCorpusCache:
    def setup_method(self) -> None:
        corpus_cache.clear()

    def test_loads_once_until_cleared(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "RETRIEVAL_CORPUS_CACHE_TTL", 900)
        loader = AsyncMock(return_value=_corpus())

        async def go():
            a = await corpus_cache.get_or_load("r1", "m", loader)
            b = await corpus_cache.get_or_load("r1", "m", loader)
            return a, b

        a, b = asyncio.run(go())
        assert a is b and loader.await_count == 1
        corpus_cache.clear()
        asyncio.run(corpus_cache.get_or_load("r1", "m", loader))
        assert loader.await_count == 2

    def test_ttl_zero_disables(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "RETRIEVAL_CORPUS_CACHE_TTL", 0)
        loader = AsyncMock(return_value=_corpus())
        asyncio.run(corpus_cache.get_or_load("r1", None, loader))
        asyncio.run(corpus_cache.get_or_load("r1", None, loader))
        assert loader.await_count == 2

    def test_empty_corpus_not_cached(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "RETRIEVAL_CORPUS_CACHE_TTL", 900)
        loader = AsyncMock(return_value=RunCorpus([], [], run_id="r1"))
        asyncio.run(corpus_cache.get_or_load("r1", None, loader))
        asyncio.run(corpus_cache.get_or_load("r1", None, loader))
        assert loader.await_count == 2

    def test_model_is_part_of_the_key(self, monkeypatch) -> None:
        monkeypatch.setattr(settings, "RETRIEVAL_CORPUS_CACHE_TTL", 900)
        loader = AsyncMock(return_value=_corpus())
        asyncio.run(corpus_cache.get_or_load("r1", None, loader))
        asyncio.run(corpus_cache.get_or_load("r1", "bge", loader))
        assert loader.await_count == 2


# ------------------------------------------------------------------ #
# API: filters + /retrieve/multi
# ------------------------------------------------------------------ #


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def serve_sample(monkeypatch):
    async def fake_build(run_id: str, with_dense: bool) -> RetrievalService:
        noise = _chunk("ref-1", "paper-001", SectionType.REFERENCES, REFERENCE_LIST)
        return RetrievalService(_corpus([noise]), FakeEmbeddingProvider())

    monkeypatch.setattr(research_routes, "_build_retrieval_service", fake_build)


class TestRetrieveApi:
    URL = f"{settings.API_V1_PREFIX}/research/retrieve"

    @pytest.mark.parametrize("method", ["sparse", "dense", "hybrid"])
    def test_gap_evidence_preset_over_http(self, client, serve_sample, method) -> None:
        resp = client.post(f"{self.URL}/{method}", json={
            "run_id": "r1", "query": "limitations of language models", "top_k": 5,
            "section_types": ["gap_evidence"],
        })
        assert resp.status_code == 200
        types = {r["section_type"] for r in resp.json()["results"]}
        assert types and types <= {"LIMITATIONS", "FUTURE_WORK", "DISCUSSION", "CONCLUSION", "RESULTS"}

    def test_references_excluded_unless_asked(self, client, serve_sample) -> None:
        body = {"run_id": "r1", "query": "language model attention transformer", "top_k": 20}
        default = client.post(f"{self.URL}/sparse", json=body).json()["results"]
        with_noise = client.post(f"{self.URL}/sparse", json={**body, "include_noise": True}).json()["results"]
        assert "REFERENCES" not in {r["section_type"] for r in default}
        assert "REFERENCES" in {r["section_type"] for r in with_noise}

    def test_bad_year_range_is_400(self, client, serve_sample) -> None:
        resp = client.post(f"{self.URL}/sparse", json={
            "run_id": "r1", "query": "x", "year_from": 2024, "year_to": 2020})
        assert resp.status_code == 400

    def test_multi_with_explicit_queries(self, client, serve_sample) -> None:
        resp = client.post(f"{self.URL}/multi", json={
            "run_id": "r1", "queries": ["transformer attention", "language model limitations"],
            "method": "hybrid", "top_k": 5})
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert results and all(r["retrieval_method"] == "multi_hybrid" for r in results)
        assert all(r["metadata"]["matched_queries"] for r in results)

    def test_multi_with_query_id_uses_question_and_expansions(self, client, serve_sample, monkeypatch) -> None:
        monkeypatch.setattr(research_routes, "_load_query_run", AsyncMock(return_value={
            "research_question": "transformer attention",
            "expanded_queries": ["language model limitations", "Transformer Attention"],
        }))
        resp = client.post(f"{self.URL}/multi", json={"run_id": "r1", "query_id": "q-1", "method": "sparse"})
        assert resp.status_code == 200
        assert resp.json()["query"] == "transformer attention | language model limitations"

    def test_multi_unknown_query_id_is_404(self, client, serve_sample, monkeypatch) -> None:
        monkeypatch.setattr(research_routes, "_load_query_run", AsyncMock(return_value=None))
        resp = client.post(f"{self.URL}/multi", json={"run_id": "r1", "query_id": "nope"})
        assert resp.status_code == 404

    def test_multi_needs_queries(self, client, serve_sample) -> None:
        assert client.post(f"{self.URL}/multi", json={"run_id": "r1"}).status_code == 400
        assert client.post(f"{self.URL}/multi", json={"run_id": "r1", "queries": ["x"], "method": "bm25"}).status_code == 400


# ------------------------------------------------------------------ #
# A1 /query -> /collect
# ------------------------------------------------------------------ #


class _FakeCollectionService:
    def __init__(self, capped: bool = False) -> None:
        self.calls: list[dict] = []
        self._capped = capped

    def collect(self, rq, max_results_per_query=None, max_total_results=None):
        self.calls.append({"queries": list(rq.expanded_queries), "per_query": max_results_per_query,
                           "total": max_total_results})
        return CollectionResult(query_id=rq.query_id, queries_searched=list(rq.expanded_queries),
                                providers_used=["fake"], papers=[], status="failed", capped=self._capped)


@pytest.fixture
def collect_env(monkeypatch):
    service = _FakeCollectionService()
    repo = MagicMock()
    repo.save_collection_result = AsyncMock()
    import app.ingestion.deduplication.service as dedup_mod
    import app.storage.relational.database as database
    import app.storage.repositories.paper_repository as paper_repo_mod

    monkeypatch.setattr(research_routes, "_get_collection_service", lambda: service)
    monkeypatch.setattr(database, "get_db_pool", AsyncMock(return_value=object()))
    monkeypatch.setattr(paper_repo_mod, "PaperRepository", MagicMock(return_value=repo))
    dedup = MagicMock()
    dedup.run = AsyncMock(return_value={})
    monkeypatch.setattr(dedup_mod, "Deduplicator", MagicMock(return_value=dedup))
    return service, repo


class TestCollectApi:
    URL = f"{settings.API_V1_PREFIX}/research/collect"

    def test_query_id_runs_question_plus_all_expansions(self, client, collect_env, monkeypatch) -> None:
        service, repo = collect_env
        monkeypatch.setattr(research_routes, "_load_query_run", AsyncMock(return_value={
            "research_question": "Gaps in retinal disease detection?",
            "expanded_queries": ["retinal disease detection", "fundus image deep learning",
                                 "Retinal Disease Detection"],
        }))
        resp = client.post(self.URL, json={"query_id": "q-1", "max_results_per_query": 7,
                                           "max_total_results": 50})
        assert resp.status_code == 200
        body = resp.json()
        assert service.calls[0]["queries"] == [
            "Gaps in retinal disease detection?", "retinal disease detection", "fundus image deep learning"]
        assert (service.calls[0]["per_query"], service.calls[0]["total"]) == (7, 50)
        assert body["query_id"] == "q-1" and body["run_id"]
        _, question = repo.save_collection_result.await_args.args
        assert question == "Gaps in retinal disease detection?"
        assert repo.save_collection_result.await_args.kwargs["query_id"] == "q-1"

    def test_include_original_false(self, client, collect_env) -> None:
        service, _ = collect_env
        client.post(self.URL, json={"query": "Q?", "expanded_queries": ["a", "b"],
                                    "include_original_query": False})
        assert service.calls[0]["queries"] == ["a", "b"]

    def test_plain_query_still_works(self, client, collect_env) -> None:
        service, _ = collect_env
        assert client.post(self.URL, json={"query": "Q?"}).status_code == 200
        assert service.calls[0]["queries"] == ["Q?"]

    def test_capped_is_reported(self, client, collect_env, monkeypatch) -> None:
        monkeypatch.setattr(research_routes, "_get_collection_service", lambda: _FakeCollectionService(capped=True))
        assert client.post(self.URL, json={"query": "Q?"}).json()["capped"] is True

    def test_errors(self, client, collect_env, monkeypatch) -> None:
        assert client.post(self.URL, json={}).status_code == 400
        assert client.post(self.URL, json={"query": "Q?", "max_total_results": 0}).status_code == 400
        monkeypatch.setattr(research_routes, "_load_query_run", AsyncMock(return_value=None))
        assert client.post(self.URL, json={"query_id": "missing"}).status_code == 404

    def test_save_failure_is_503_without_run_id(self, client, collect_env) -> None:
        _, repo = collect_env
        repo.save_collection_result.side_effect = RuntimeError("db down")
        resp = client.post(self.URL, json={"query": "Q?"})
        assert resp.status_code == 503
        assert "run_id" not in resp.json()


class TestQueryApi:
    def test_stub_query_is_saved_and_returns_query_id(self, client, monkeypatch) -> None:
        saved = []

        async def fake_save(q):
            saved.append(q)
            return True

        monkeypatch.setattr(research_routes, "_save_query_run", fake_save)
        resp = client.post(f"{settings.API_V1_PREFIX}/research/query", json={"query": "Retinal gaps?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["saved"] is True
        assert saved and saved[0].query_id == body["query_id"] and saved[0].query == "Retinal gaps?"


# ------------------------------------------------------------------ #
# Collection service: provider interleaving, overrides, capped
# ------------------------------------------------------------------ #


class _Provider:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, limit: int):
        from app.domain.models.collection import CollectedPaper, PaperProvenance
        from app.domain.models.paper import Paper

        self.calls.append((query, limit))
        return [
            CollectedPaper(
                paper=Paper(title=f"{self.name} {query} {i}"),
                provenance=PaperProvenance(source=self.name, source_query=query, provider_id=f"{self.name}{i}"),
            )
            for i in range(limit)
        ]


class TestCollectionService:
    def test_every_provider_searches_each_query_before_the_cap(self) -> None:
        from app.domain.models.query import ResearchQuery

        a, b = _Provider("a"), _Provider("b")
        service = LiteratureCollectionService({"a": a, "b": b}, max_results_per_query=5, max_total_results=200)
        result = service.collect(ResearchQuery(query="q", expanded_queries=["q1", "q2", "q3"]),
                                 max_results_per_query=3, max_total_results=13)
        assert a.calls[0] == ("q1", 3) and b.calls[0] == ("q1", 3)
        assert len(result.papers) == 13
        assert result.capped is True
        assert {"a", "b"} == set(result.providers_used)

    def test_not_capped_when_budget_suffices(self) -> None:
        from app.domain.models.query import ResearchQuery

        service = LiteratureCollectionService({"a": _Provider("a")}, max_results_per_query=2, max_total_results=100)
        result = service.collect(ResearchQuery(query="q", expanded_queries=["q1", "q2"]))
        assert len(result.papers) == 4 and result.capped is False


# ------------------------------------------------------------------ #
# Heading detector: journal page headers
# ------------------------------------------------------------------ #


@pytest.mark.parametrize("line", [
    "Volume 8, Issue 2, March-April 2026",
    "IJFMR260270709",
    "CC-BY 4.0 International license",
])
def test_page_headers_are_not_headings(line: str) -> None:
    lines = [(0, 1, ""), (1, 1, line), (2, 1, ""),
             (3, 1, "This is a long paragraph line with many words in it to follow headings.")]
    assert detect_headings(lines) == []
