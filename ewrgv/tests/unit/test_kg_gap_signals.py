"""
tests/unit/test_kg_gap_signals.py
-----------------------------------
Knowledge-graph build, D1 explicit statements, D2 combination sparsity,
fusion / GapScore v1, ranking, grounding gate.  Uses a bag-of-words fake
embedder so similarities are controlled and nothing is downloaded.
"""

from __future__ import annotations

import asyncio
import math
import re

import pytest

from app.domain.enums import SectionType
from app.domain.models.paper import Paper
from app.domain.models.paper_chunk import PaperChunk
from app.gap_detection.signals import fusion
from app.gap_detection.signals.combination import detect_combinations
from app.gap_detection.signals.explicit import detect_explicit
from app.gap_detection.signals.service import GapDetectionService
from app.ingestion.corpus.run_corpus import RunCorpus
from app.knowledge_graph.builder import KnowledgeGraphBuilder
from app.knowledge_graph.graph import KnowledgeGraph

VOCAB = [
    "generalisation", "datasets", "unseen", "hospital", "oct", "fundus", "imbalance", "class", "label", "labelled",
    "annotation", "semi", "supervised", "transformer", "vision", "cnn", "glaucoma", "retinopathy", "diabetic",
    "screening", "segmentation", "lesion", "small", "retinal", "disease", "detection", "deep", "learning",
    "clinical", "interpretability", "transparency", "black", "box", "plant", "crop", "leaf", "wheat",
]


class BowEmbedder:
    """Deterministic bag-of-words vectors: shared words -> high cosine."""

    def get_model_name(self) -> str:
        return "bow-test"

    def get_dimension(self) -> int:
        return len(VOCAB) + 1

    def embed_batch(self, texts):
        out = []
        for t in texts:
            words = re.findall(r"[a-z]+", t.lower())
            v = [float(sum(1 for w in words if w.startswith(term))) for term in VOCAB] + [0.05]
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out

    def embed(self, text):
        return self.embed_batch([text])[0]


GEN = ("Generalisation of deep learning to unseen hospital datasets remains largely unexplored for "
       "retinal disease detection with OCT.")
GEN2 = ("Most models are limited to a single hospital and generalisation to unseen OCT datasets is poor in "
        "retinal disease detection.")
IMB = "Class imbalance in retinal datasets remains an open problem for glaucoma and retinopathy screening tasks."
IMB2 = "Class imbalance is a key limitation; models struggle with rare retinopathy and glaucoma classes in screening."


def _paper(pid: str, title: str, year: int) -> Paper:
    return Paper(paper_id=pid, title=title, abstract=title, publication_year=year)


def _chunk(cid: str, pid: str, stype: SectionType, text: str, page: int = 3) -> PaperChunk:
    return PaperChunk(chunk_id=cid, paper_id=pid, section_id=f"s-{cid}", section_title=stype.value.title(),
                      section_type=stype, text=text, page_start=page, page_end=page)


def make_corpus() -> RunCorpus:
    papers = [
        _paper("p1", "Deep learning retinal disease detection OCT", 2023),
        _paper("p2", "Retinal disease detection with CNN fundus", 2024),
        _paper("p3", "Vision transformer retinal disease detection OCT", 2024),
        _paper("p4", "Glaucoma retinopathy screening deep learning", 2025),
        _paper("p5", "Fundus retinal disease detection screening", 2025),
        _paper("px", "Wheat leaf crop plant yields", 2025),
    ]
    method = "We train a CNN and a vision transformer on fundus images and OCT scans for diabetic retinopathy."
    chunks = [
        _chunk("c1", "p1", SectionType.LIMITATIONS, GEN + " " + method),
        _chunk("c2", "p2", SectionType.CONCLUSION, GEN2 + " " + IMB),
        _chunk("c3", "p3", SectionType.DISCUSSION, IMB2 + " Vision transformer on OCT scans for glaucoma."),
        _chunk("c4", "p4", SectionType.METHODS, "A CNN on fundus images for glaucoma screening and diabetic retinopathy."),
        _chunk("c5", "p5", SectionType.RESULTS, "A CNN on fundus images for diabetic retinopathy screening with high accuracy."),
        _chunk("c6", "p4", SectionType.FUTURE_WORK, IMB + " Future work should address this."),
        _chunk("cx", "px", SectionType.LIMITATIONS, "Wheat leaf crop plant disease detection remains largely unexplored in low light fields."),
    ]
    return RunCorpus(papers, chunks, run_id="r1")


QUESTION = "research gaps in retinal disease detection with deep learning"


def build(corpus=None, **kw) -> KnowledgeGraph:
    return KnowledgeGraphBuilder(BowEmbedder(), **kw).build(
        corpus or make_corpus(), "r1", QUESTION, ["retinal disease detection deep learning"],
        {"methods": [], "concepts": ["retinal disease"], "domain": "ophthalmology"}, "q1")


# ------------------------------------------------------------------ #
# Knowledge graph
# ------------------------------------------------------------------ #


class TestBuilder:
    def test_relevance_gate_marks_off_topic(self) -> None:
        kg = build(relevance_threshold=0.3)
        assert kg.papers["px"]["on_topic"] is False
        assert all(kg.papers[p]["on_topic"] for p in ("p1", "p2", "p3"))
        assert not any(s["paper_id"] == "px" for s in kg.statements)

    def test_entities_mentions_cooccurrence(self) -> None:
        kg = build(relevance_threshold=0.3)
        assert "METHOD:convolutional neural network" in kg.entities
        assert "MODALITY:optical coherence tomography" in kg.entities
        cnn = kg.entities["METHOD:convolutional neural network"]
        assert set(cnn["papers"]) >= {"p1", "p4", "p5"}
        assert "CNN" in cnn["aliases"]
        edge = next(e for e in kg.cooccurrence if {e["a"], e["b"]} == {"METHOD:convolutional neural network", "MODALITY:fundus photography"})
        assert edge["papers"] >= 2
        assert kg.stats["gap_statements"] >= 4

    def test_query_terms_flagged(self) -> None:
        kg = build(relevance_threshold=0.3)
        assert kg.entities["CONCEPT:retinal disease"]["query_term"] is True

    def test_statements_deduplicated_and_typed(self) -> None:
        corpus = make_corpus()
        corpus = RunCorpus(corpus.get_all_papers(),
                           corpus.get_all_chunks() + [_chunk("c1b", "p1", SectionType.LIMITATIONS, GEN)], run_id="r1")
        kg = build(corpus, relevance_threshold=0.3)
        assert sum(1 for s in kg.statements if s["sentence"] == GEN) == 1
        assert {s["cue_class"] for s in kg.statements} >= {"OPEN_PROBLEM", "LIMITATION"}

    def test_payload_round_trip_and_exports(self) -> None:
        kg = build(relevance_threshold=0.3)
        again = KnowledgeGraph.from_payload(kg.to_payload())
        assert again.stats == kg.stats and again.entities == kg.entities
        cy = kg.to_cytoscape()
        kinds = {n["data"]["kind"] for n in cy["elements"]["nodes"]}
        assert {"Paper", "Entity", "GapStatement"} <= kinds
        assert kg.to_graphml().startswith("<?xml")

    def test_curated_entities_never_merge(self) -> None:
        class Same(BowEmbedder):
            def embed_batch(self, texts):
                return [[1.0, 0.0] for _ in texts]  # everything identical

        kg = KnowledgeGraphBuilder(Same(), relevance_threshold=0.0).build(make_corpus(), "r1", QUESTION)
        assert "METRIC:accuracy" in kg.entities or "TASK:screening" in kg.entities
        assert "METHOD:convolutional neural network" in kg.entities and "METHOD:vision transformer" in kg.entities


# ------------------------------------------------------------------ #
# Detectors
# ------------------------------------------------------------------ #


class TestDetectors:
    def test_d1_groups_the_same_gap_across_papers(self) -> None:
        kg = build(relevance_threshold=0.3)
        cands = detect_explicit(kg, BowEmbedder(), similarity=0.6)
        gen = next(c for c in cands if "generalisation" in c.statement.lower())
        assert gen.signals["stating_papers"] >= 2
        assert 0 < gen.signals["explicit"] < 1
        imb = next(c for c in cands if "imbalance" in c.statement.lower())
        assert {s["paper_id"] for s in imb.statements} >= {"p2", "p3", "p4"}

    def test_d1_skips_statements_without_specific_concepts(self) -> None:
        kg = build(relevance_threshold=0.3)
        for s in kg.statements:
            s["entities"] = []
        assert detect_explicit(kg, BowEmbedder()) == []

    def test_d2_finds_sparse_but_linked_pairs(self) -> None:
        kg = KnowledgeGraph(run_id="r", query_id=None)
        pa, pb = [f"a{i}" for i in range(6)], [f"b{i}" for i in range(6)]
        kg.papers = {p: {"title": p, "on_topic": True, "year": 2024} for p in pa + pb}
        hub = sorted(pa + pb)
        kg.entities = {
            "METHOD:m": {"type": "METHOD", "name": "m", "papers": pa, "aliases": []},
            "MODALITY:x": {"type": "MODALITY", "name": "x", "papers": pb, "aliases": []},
            "TASK:t": {"type": "TASK", "name": "t", "papers": hub, "aliases": []},
            "DISEASE:d": {"type": "DISEASE", "name": "d", "papers": hub, "aliases": []},
        }
        kg.cooccurrence = [
            {"a": "METHOD:m", "b": "TASK:t", "papers": 6, "pmi": 0.5},
            {"a": "MODALITY:x", "b": "TASK:t", "papers": 6, "pmi": 0.5},
            {"a": "METHOD:m", "b": "DISEASE:d", "papers": 6, "pmi": 0.5},
            {"a": "MODALITY:x", "b": "DISEASE:d", "papers": 6, "pmi": 0.5},
        ]
        found = detect_combinations(kg, min_papers=3)
        pair = next(c for c in found if set(c.core_entities) == {"METHOD:m", "MODALITY:x"})
        assert pair.signals["papers_both"] == 0
        assert pair.signals["sparsity"] == 1.0
        assert pair.signals["link_prediction"] == 1.0  # same neighbours


# ------------------------------------------------------------------ #
# Fusion, scoring, ranking, grounding
# ------------------------------------------------------------------ #


class TestFusionAndService:
    def test_score_breakdown_sums_and_no_llm_confidence(self) -> None:
        kg = build(relevance_threshold=0.3)
        result = asyncio.run(GapDetectionService(kg, make_corpus(), BowEmbedder()).run())
        assert result["candidates"]
        for c in result["candidates"]:
            parts = sum(v["contribution"] for v in c["score_breakdown"].values())
            assert c["gap_score"] == pytest.approx(min(1.0, parts), abs=1e-3)
            assert set(c["score_breakdown"]) == {"explicit", "sparsity", "independence", "agreement_bonus"}
            assert "initial_confidence" not in c and "llm_confidence" not in c
        assert [c["rank"] for c in result["candidates"]] == list(range(1, len(result["candidates"]) + 1))

    def test_multi_paper_gap_outranks_single_paper(self) -> None:
        kg = build(relevance_threshold=0.3)
        result = asyncio.run(GapDetectionService(kg, make_corpus(), BowEmbedder()).run(detectors=["explicit"]))
        top = result["candidates"][0]
        assert top["score_notes"]["stating_papers"] >= 2
        assert top["supporting_statements"][0]["section_type"] in {"LIMITATIONS", "CONCLUSION", "DISCUSSION", "FUTURE_WORK"}
        assert top["evidence_passages"][0]["page_start"] == 3

    def test_small_corpus_discounts_sparsity(self) -> None:
        kg = build(relevance_threshold=0.3)
        group = {"primary": type("C", (), {"statements": [], "core_entities": [], "detector": "combination_sparsity",
                                           "signals": {"sparsity": 1.0, "link_prediction": 1.0, "lift": 0.0,
                                                       "papers_both": 0, "expected": 2.0}})(),
                 "detectors": ["combination_sparsity"], "extra": []}
        scored = fusion.score(kg, group)
        assert scored["score_notes"]["sparsity_reliability"] < 1
        assert scored["score_breakdown"]["sparsity"]["value"] == pytest.approx(scored["score_notes"]["sparsity_reliability"])

    def test_rank_diversity_penalty_is_capped(self) -> None:
        cands = [{"gap_score": 0.9 - i * 0.01, "gap_type": "Methodological", "gap_statement": f"m{i}"} for i in range(5)]
        cands.append({"gap_score": 0.6, "gap_type": "Dataset", "gap_statement": "d"})
        ranked = fusion.rank(cands)
        assert ranked[0]["gap_statement"] == "m0"
        assert max(c["gap_score"] - c["rank_score"] for c in ranked) == pytest.approx(2 * fusion.DIVERSITY_PENALTY)

    def test_grounding_marks_uncertain_when_statement_drifts(self) -> None:
        kg = build(relevance_threshold=0.3)
        service = GapDetectionService(kg, make_corpus(), BowEmbedder())
        result = asyncio.run(service.run(detectors=["explicit"]))
        c = dict(result["candidates"][0])
        c["gap_statement"] = "Wheat leaf crop plant yields in fields"
        c["status"] = "candidate"
        asyncio.run(service._ground(c))
        assert c["grounding"]["grounded"] is False and c["status"] == "uncertain"

    def test_gap_type_rules(self) -> None:
        kg = build(relevance_threshold=0.3)
        assert fusion.gap_type_for(kg, [], "Class imbalance in the datasets limits training").value == "Dataset"
        assert fusion.gap_type_for(kg, [], "Models are rarely studied in rural low-resource settings").value == "Population"
        assert fusion.gap_type_for(kg, [], "External validation is lacking").value == "Evaluation"
        assert fusion.gap_type_for(kg, [], "Attention is rarely combined with recurrence").value == "Methodological"
