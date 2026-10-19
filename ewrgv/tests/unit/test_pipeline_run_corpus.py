"""
tests/unit/test_pipeline_run_corpus.py
----------------------------------------
Pipeline Stages 1 → 7 over a run's processed corpus (RunCorpus of
full-text PaperChunks), offline.

Real QueryUnderstandingService, RunCorpus, BM25, GapDetector,
CandidateEvidenceScorer and CandidateRanker; the collection service returns
a hand-created CollectionResult, the corpus loader returns an in-memory
RunCorpus, and the LLM is the strict ScriptedFakeLLM.  No API key, network
or database.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.exceptions import IngestionError, RetrievalError
from app.domain.enums import PipelineStage
from app.domain.models.paper_chunk import PaperChunk
from app.gap_detection.detector import GapDetector
from app.gap_detection.evidence_scoring import CandidateEvidenceScorer
from app.gap_detection.ranking import CandidateRanker
from app.ingestion.collectors.service import LiteratureCollectionService
from app.ingestion.corpus.run_corpus import RunCorpus
from app.orchestration.research_pipeline import (
    CORPUS_SOURCE_INJECTED,
    CORPUS_SOURCE_RUN,
    ResearchPipeline,
)
from app.query_understanding.service import QueryUnderstandingService
from app.retrieval.service import RetrievalService
from tests.fixtures.fake_llm import ScriptedFakeLLM
from tests.fixtures.sample_collection import (
    CAN_GNN_CHUNK,
    GRID_CHUNK,
    QUERY_ID,
    Q_EXPANDED,
    TRANSFER_CHUNK,
    make_collection_result,
    make_run_chunks,
    make_run_papers,
)
from tests.fixtures.sample_corpus import FakeEmbeddingProvider


def _understanding() -> dict:
    return {
        "domain": "Automotive Cybersecurity",
        "problem": "CAN bus attack detection",
        "task": "intrusion detection",
        "methods": ["Graph Neural Networks"],
        "concepts": ["CAN bus"],
        "entities": [], "datasets": [], "metrics": [], "constraints": [],
        "research_intent": "gap identification",
        "terminology_variants": [], "method_variants": [],
        "expanded_queries": [Q_EXPANDED],
    }


def _detection() -> dict:
    return {
        "candidate_gaps": [
            {
                "gap_statement": "GNN-based CAN intrusion detectors are not evaluated on "
                                 "real-time embedded hardware.",
                "gap_type": "Evaluation",
                "supporting_chunk_ids": [CAN_GNN_CHUNK],
                "initial_confidence": 0.8,
                "key_concepts": ["graph neural network", "CAN bus", "embedded hardware"],
            },
            {
                "gap_statement": "CAN intrusion detectors lack evaluation across vehicle "
                                 "manufacturers on real traffic.",
                "gap_type": "Dataset",
                "supporting_chunk_ids": [f"[chunk_id: {TRANSFER_CHUNK}]", CAN_GNN_CHUNK],
                "initial_confidence": 0.7,
            },
            {   # not in this run's corpus → ungrounded, dropped
                "gap_statement": "GPT-3 is weak at structured prediction.",
                "gap_type": "Methodological",
                "supporting_chunk_ids": ["chunk-003-limits"],
                "initial_confidence": 0.9,
            },
        ]
    }


def _llm() -> ScriptedFakeLLM:
    return ScriptedFakeLLM(_understanding(), _detection())


def _corpus(chunks=None) -> RunCorpus:
    return RunCorpus(
        make_run_papers(), make_run_chunks() if chunks is None else chunks, run_id=QUERY_ID
    )


def _forbidden_retriever() -> MagicMock:
    retriever = MagicMock()
    retriever.retrieve.side_effect = AssertionError("injected retriever must not be used")
    return retriever


def _collection_service() -> MagicMock:
    service = MagicMock(spec=LiteratureCollectionService)
    service.collect.return_value = make_collection_result()
    return service


def _pipeline(llm, collection_service=None, retriever=None, **kwargs) -> ResearchPipeline:
    return ResearchPipeline(
        query_understanding_service=QueryUnderstandingService(llm=llm),
        literature_collector=None,
        document_parser=None,
        retriever=retriever or _forbidden_retriever(),
        reranker=None,
        gap_detector=GapDetector(llm),
        candidate_ranker=CandidateRanker(),
        ewrgv_validator=None,
        literature_collection_service=collection_service,
        **kwargs,
    )


# ======================================================================
# Collection → corpus_loader → detection
# ======================================================================


class TestRunCorpusPath:
    def _run(self, **kwargs):
        loader = MagicMock(return_value=_corpus())
        llm = _llm()
        result = _pipeline(
            llm, _collection_service(), corpus_loader=loader, **kwargs
        ).run("CAN bus intrusion detection gaps")
        return result, llm, loader

    def test_loader_called_with_run_id(self):
        result, _, loader = self._run()
        loader.assert_called_once_with(QUERY_ID)
        assert result.corpus_source == CORPUS_SOURCE_RUN
        assert result.stage_reached == PipelineStage.CANDIDATE_RANKING
        assert result.corpus_stats["chunks"] == 4

    def test_detector_sees_canonical_paper_chunks(self):
        result, _, _ = self._run()
        corpus_ids = {c.chunk_id for c in make_run_chunks()}
        assert result.retrieved_chunks
        assert all(isinstance(c, PaperChunk) for c in result.retrieved_chunks)
        assert {c.chunk_id for c in result.retrieved_chunks} <= corpus_ids
        assert GRID_CHUNK not in {c.chunk_id for c in result.retrieved_chunks}

    def test_retrieval_provenance_recorded(self):
        result, _, _ = self._run()
        prov = result.retrieval_provenance[CAN_GNN_CHUNK]
        assert prov["retrieval_method"] == "sparse"
        assert prov["retrieval_rank"] >= 1
        assert prov["query"]

    def test_prompt_shows_section_types(self):
        _, llm, _ = self._run()
        prompt = next(p for kind, p in llm.calls if kind == "detection")
        assert f"[chunk_id: {CAN_GNN_CHUNK}]" in prompt
        assert "section: 5 Limitations [LIMITATIONS]" in prompt

    def test_grounding_to_run_corpus(self):
        result, _, _ = self._run()
        statements = [g.gap_statement for g in result.candidate_gaps]
        assert len(statements) == 2
        assert not any("GPT-3" in s for s in statements)
        by_type = {g.primary_gap_type.value: g for g in result.candidate_gaps}
        assert set(by_type["Dataset"].source_papers) == {"oa-transfer", "s2-can-gnn"}

    def test_evidence_scorer_uses_corpus_papers(self):
        result, _, _ = self._run(evidence_scorer=CandidateEvidenceScorer())
        assert result.stage_reached == PipelineStage.CANDIDATE_RANKING
        for gap in result.top_k_gaps:
            for item in gap.evidence_assessment.supporting:
                assert item.publication_year in (2022, 2023)  # from the Paper rows
                assert item.title
                assert item.section_type in ("LIMITATIONS", "FUTURE_WORK")
                assert item.page_start >= 1
                assert item.quality == 1.0  # explicit-gap section

    def test_hybrid_retriever_factory(self):
        result, _, _ = self._run(
            retriever_factory=lambda corpus: RetrievalService(corpus, FakeEmbeddingProvider())
        )
        assert result.stage_reached == PipelineStage.CANDIDATE_RANKING
        assert result.retrieval_provenance[CAN_GNN_CHUNK]["retrieval_method"] == "hybrid"

    def test_explicit_corpus_skips_loader(self):
        loader = MagicMock()
        result = _pipeline(_llm(), _collection_service(), corpus_loader=loader).run(
            "q", corpus=_corpus()
        )
        loader.assert_not_called()
        assert result.stage_reached == PipelineStage.CANDIDATE_RANKING

    def test_run_on_corpus(self):
        from app.domain.models.query import ResearchQuery

        query = ResearchQuery(query="CAN bus intrusion detection", expanded_queries=[Q_EXPANDED])
        result = _pipeline(_llm()).run_on_corpus(query, _corpus())
        assert result.corpus_source == CORPUS_SOURCE_RUN
        assert len(result.candidate_gaps) == 2


# ======================================================================
# Missing / empty corpus and failures
# ======================================================================


class TestMissingCorpus:
    def test_collection_without_corpus_stops_with_instructions(self):
        result = _pipeline(_llm(), _collection_service()).run("q")
        assert result.stage_reached == PipelineStage.LITERATURE_COLLECTION
        assert "/chunks" in result.error and "/represent" in result.error
        assert result.candidate_gaps == []

    def test_empty_corpus_stops_at_document_processing(self):
        loader = MagicMock(return_value=_corpus(chunks=[]))
        result = _pipeline(_llm(), _collection_service(), corpus_loader=loader).run("q")
        assert result.stage_reached == PipelineStage.DOCUMENT_PROCESSING
        assert "no chunks" in result.error
        assert "3 papers" in result.error

    def test_loader_error_wrapped(self):
        loader = MagicMock(side_effect=OSError("storage down"))
        result = _pipeline(_llm(), _collection_service(), corpus_loader=loader).run("q")
        assert result.stage_reached == PipelineStage.FAILED
        assert "Corpus loading failed" in result.error and "storage down" in result.error

    def test_loader_ewrgv_error_message_kept(self):
        loader = MagicMock(side_effect=IngestionError("run not found"))
        result = _pipeline(_llm(), _collection_service(), corpus_loader=loader).run("q")
        assert result.stage_reached == PipelineStage.FAILED
        assert result.error == "run not found"

    def test_retriever_factory_error_wrapped(self):
        def factory(corpus):
            raise ValueError("bad index")

        result = _pipeline(_llm(), retriever_factory=factory).run("q", corpus=_corpus())
        assert result.stage_reached == PipelineStage.FAILED
        assert "Retriever construction failed" in result.error
        assert result.corpus_source == CORPUS_SOURCE_RUN

    def test_retrieval_error_on_corpus_path(self):
        class FailingRetriever:
            def retrieve(self, query, top_k):
                raise RetrievalError("bm25 broken")

        result = _pipeline(_llm(), retriever_factory=lambda c: FailingRetriever()).run(
            "q", corpus=_corpus()
        )
        assert result.stage_reached == PipelineStage.FAILED
        assert result.error == "bm25 broken"

    def test_results_outside_corpus_dropped(self):
        from app.domain.models.retrieval_result import RetrievalResult

        class Retriever:
            def retrieve(self, query, top_k):
                return [
                    RetrievalResult(chunk_id="ghost", paper_id="x", text="t", score=1.0,
                                    rank=1, retrieval_method="sparse"),
                    RetrievalResult(chunk_id=CAN_GNN_CHUNK, paper_id="s2-can-gnn", text="t",
                                    score=0.5, rank=2, retrieval_method="sparse"),
                ]

        result = _pipeline(_llm(), retriever_factory=lambda c: Retriever()).run(
            "q", corpus=_corpus()
        )
        assert [c.chunk_id for c in result.retrieved_chunks] == [CAN_GNN_CHUNK]
        # The corpus's chunk, not one rebuilt from the RetrievalResult text.
        assert result.retrieved_chunks[0].section_title == "5 Limitations"


class TestInjectedRetrieverWithoutCollection:
    def test_injected_retriever_used_when_no_collection_and_no_corpus(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = [make_run_chunks()[0]]
        result = _pipeline(_llm(), retriever=retriever).run("q")
        assert result.corpus_source == CORPUS_SOURCE_INJECTED
        assert result.stage_reached == PipelineStage.CANDIDATE_RANKING
        assert [c.chunk_id for c in result.retrieved_chunks] == [CAN_GNN_CHUNK]


@pytest.mark.parametrize("top_k", [1, 2])
def test_retrieval_top_k_caps_corpus_results(top_k):
    result = _pipeline(_llm(), retrieval_top_k=top_k).run("q", corpus=_corpus())
    assert len(result.retrieved_chunks) <= top_k
