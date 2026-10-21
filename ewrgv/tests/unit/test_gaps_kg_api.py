"""
tests/unit/test_gaps_kg_api.py
--------------------------------
/research/kg/* and /research/gaps/* endpoints with mocked storage and jobs.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

import app.api.routes.gaps_kg as routes
import app.knowledge_graph.jobs as jobs
from app.core.config import settings
from app.gap_detection.signals.service import GapDetectionService
from app.main import app
from tests.unit.test_kg_gap_signals import BowEmbedder, build, make_corpus

BASE = f"{settings.API_V1_PREFIX}/research"


@pytest.fixture
def stored():
    kg = build(relevance_threshold=0.3)
    result = asyncio.run(GapDetectionService(kg, make_corpus(), BowEmbedder()).run())
    return {
        "graph": {"run_id": "r1", "query_id": "q1", "built_at": "2026-10-19T00:00:00", "stats": kg.stats,
                  "payload": kg.to_payload()},
        "gaps": {"run_id": "r1", "query_id": "q1", "created_at": "2026-10-19T00:00:00",
                 "params": {"detectors": ["explicit_statements", "combination_sparsity"]},
                 "stats": result["stats"], "candidates": result["candidates"], "baseline": None},
    }


@pytest.fixture
def env(monkeypatch, stored):
    import app.storage.relational.database as database
    import app.storage.repositories.kg_repository as kg_repo_mod
    import app.storage.repositories.paper_repository as paper_repo_mod

    repo = MagicMock()
    repo.get_graph = AsyncMock(side_effect=lambda rid: stored["graph"] if rid == "r1" else None)
    repo.get_gaps = AsyncMock(side_effect=lambda rid: stored["gaps"] if rid == "r1" else None)
    papers = MagicMock()
    papers.run_exists = AsyncMock(side_effect=lambda rid: rid == "r1")
    monkeypatch.setattr(database, "get_db_pool", AsyncMock(return_value=object()))
    monkeypatch.setattr(kg_repo_mod, "KnowledgeGraphRepository", MagicMock(return_value=repo))
    monkeypatch.setattr(paper_repo_mod, "PaperRepository", MagicMock(return_value=papers))

    calls = {}

    async def fake_kg_build(job, store, query_id=None, relevance_threshold=None):
        calls["kg"] = (query_id, relevance_threshold)
        from app.core.job_store import JobStatus
        job.status, job.progress, job.summary = JobStatus.COMPLETED, "done", {"entities": 3}
        await store.update_analysis_job(job)

    async def fake_gaps(job, store, **kw):
        calls["gaps"] = kw
        from app.core.job_store import JobStatus
        job.status, job.progress = JobStatus.COMPLETED, "done"
        await store.update_analysis_job(job)

    monkeypatch.setattr(jobs, "run_kg_build", fake_kg_build)
    monkeypatch.setattr(jobs, "run_gap_detection", fake_gaps)
    return calls


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


class TestKgEndpoints:
    def test_build_starts_job_and_status_reports_it(self, client, env) -> None:
        r = client.post(f"{BASE}/kg/build", json={"run_id": "r1", "query_id": "q1", "relevance_threshold": 0.7})
        assert r.status_code == 202
        job_id = r.json()["job_id"]
        s = client.get(f"{BASE}/kg/status/{job_id}").json()
        assert s["kind"] == "kg_build" and s["status"] == "COMPLETED" and s["summary"] == {"entities": 3}
        assert env["kg"] == ("q1", 0.7)

    def test_unknown_run_and_job(self, client, env) -> None:
        assert client.post(f"{BASE}/kg/build", json={"run_id": "nope"}).status_code == 404
        assert client.get(f"{BASE}/kg/status/missing").status_code == 404
        assert client.get(f"{BASE}/kg/nope").status_code == 404

    def test_summary_view(self, client, env) -> None:
        body = client.get(f"{BASE}/kg/r1").json()
        assert body["stats"]["entities"] > 0
        assert {"top_entities", "communities", "off_topic_papers", "strongest_cooccurrences"} <= set(body)
        assert "px" in {p["paper_id"] for p in body["off_topic_papers"]}

    @pytest.mark.parametrize("view,key", [("entities", "entities"), ("statements", "statements"),
                                          ("communities", "communities"), ("papers", "papers"),
                                          ("cytoscape", "elements")])
    def test_other_views(self, client, env, view, key) -> None:
        r = client.get(f"{BASE}/kg/r1", params={"view": view, "limit": 5})
        assert r.status_code == 200 and key in r.json()

    def test_entity_type_filter(self, client, env) -> None:
        rows = client.get(f"{BASE}/kg/r1", params={"view": "entities", "entity_type": "method"}).json()["entities"]
        assert rows and {r["type"] for r in rows} == {"METHOD"}

    def test_graphml_download(self, client, env) -> None:
        r = client.get(f"{BASE}/kg/r1", params={"view": "graphml"})
        assert r.status_code == 200 and r.headers["content-type"].startswith("application/xml")
        assert "graphml" in r.text[:300]


class TestGapEndpoints:
    def test_detect_starts_signal_job_without_llm(self, client, env) -> None:
        r = client.post(f"{BASE}/gaps", json={"run_id": "r1", "detectors": ["explicit"]})
        assert r.status_code == 202
        s = client.get(f"{BASE}/gaps/status/{r.json()['job_id']}").json()
        assert s["kind"] == "gap_detection" and s["status"] == "COMPLETED"
        assert env["gaps"]["detectors"] == ["explicit"]
        assert env["gaps"]["llm_wording"] is False
        assert env["gaps"]["include_llm_baseline"] is False

    @pytest.mark.parametrize("option", ["llm_wording", "include_llm_baseline"])
    def test_llm_modes_are_rejected(self, client, env, option) -> None:
        response = client.post(f"{BASE}/gaps", json={"run_id": "r1", option: True})
        assert response.status_code == 422

    def test_gap_api_requires_reranker_enabled(self, client, env, monkeypatch) -> None:
        monkeypatch.setattr(settings, "RERANK_ENABLED", False)
        response = client.post(f"{BASE}/gaps", json={"run_id": "r1"})
        assert response.status_code == 503

    def test_bad_detector_is_422(self, client, env) -> None:
        assert client.post(f"{BASE}/gaps", json={"run_id": "r1", "detectors": ["magic"]}).status_code == 422

    def test_summary_cards_are_readable(self, client, env) -> None:
        body = client.get(f"{BASE}/gaps/r1", params={"top": 3}).json()
        card = body["top_gaps"][0]
        assert {"gap_statement", "gap_score", "score_breakdown", "detectors_agreeing", "supporting_statements",
                "evidence_passages", "counter_hints", "grounding"} <= set(card)
        ev = card["evidence_passages"][0]
        assert {"paper", "section", "section_type", "pages", "snippet"} <= set(ev)
        assert body["stats"]["candidates_after_fusion"] >= len(body["top_gaps"])

    def test_full_compare_and_filter(self, client, env) -> None:
        full = client.get(f"{BASE}/gaps/r1", params={"view": "full", "top": 2}).json()
        assert len(full["candidates"]) <= 2 and "signals" in full["candidates"][0]
        compare = client.get(f"{BASE}/gaps/r1", params={"view": "compare"}).json()
        assert compare["llm_baseline_top"] == [] and "include_llm_baseline" in compare["note"]
        d1 = client.get(f"{BASE}/gaps/r1", params={"detector": "explicit_statements"}).json()["top_gaps"]
        assert all("explicit_statements" in c["detectors_agreeing"] for c in d1)

    def test_missing_results_404(self, client, env) -> None:
        assert client.get(f"{BASE}/gaps/nope").status_code == 404
        assert client.get(f"{BASE}/gaps/status/missing").status_code == 404
