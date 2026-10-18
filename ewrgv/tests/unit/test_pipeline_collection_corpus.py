"""
tests/unit/test_pipeline_collection_corpus.py
-----------------------------------------------
Pipeline Stages 1 → 7 over COLLECTED papers (titles + abstracts), offline.

Real QueryUnderstandingService, CollectionCorpusStore, BM25, GapDetector and
CandidateRanker; the collection service returns a hand-created
CollectionResult and the LLM is the strict ScriptedFakeLLM.  No API key,
network or database.

Also checks the fallback: without a collection result, the injected
retriever (stub corpus) is used exactly as before.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.exceptions import IngestionError, RetrievalError
from app.domain.enums import GapType, PipelineStage
from app.domain.models.collection import CollectionResult
from app.gap_detection.detector import GapDetector
from app.gap_detection.ranking import CandidateRanker
from app.ingestion.collectors.service import LiteratureCollectionService
from app.ingestion.corpus.collection_corpus import CollectionCorpusStore
from app.orchestration.research_pipeline import ResearchPipeline
from app.query_understanding.service import QueryUnderstandingService
from app.retrieval.bm25 import SparseRetriever
from app.retrieval.service import RetrievalService
from tests.fixtures.fake_llm import ScriptedFakeLLM
from tests.fixtures.sample_collection import (
    CAN_GNN_CHUNK,
    GRID_CHUNK,
    Q_EXPANDED,
    Q_MAIN,
    TRANSFER_CHUNK,
    make_collection_result,
)

COLLECTION_CHUNKS = {CAN_GNN_CHUNK, TRANSFER_CHUNK, GRID_CHUNK}


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
            {   # stub-corpus chunk: not supplied → ungrounded, dropped
                "gap_statement": "GPT-3 is weak at structured prediction.",
                "gap_type": "Methodological",
                "supporting_chunk_ids": ["chunk-003-limits"],
                "initial_confidence": 0.9,
            },
            {   # paper without abstract has no chunk → ungrounded, dropped
                "gap_statement": "Automotive security surveys omit CAN attacks.",
                "gap_type": "Other / Unknown",
                "supporting_chunk_ids": ["s2-survey:abstract"],
                "initial_confidence": 0.9,
            },
        ]
    }


def _forbidden_retriever() -> MagicMock:
    retriever = MagicMock()
    retriever.retrieve.side_effect = AssertionError("stub/injected retriever must not be used")
    return retriever


def _collection_service(result=None, raises=None) -> MagicMock:
    service = MagicMock(spec=LiteratureCollectionService)
    if raises is not None:
        service.collect.side_effect = raises
    else:
        service.collect.return_value = result
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


def _run(**kwargs):
    llm = ScriptedFakeLLM(_understanding(), _detection())
    service = _collection_service(make_collection_result())
    result = _pipeline(llm, service, **kwargs).run(Q_MAIN)
    return llm, result


class TestCollectionCorpusPath:
    def test_uses_collection_corpus_not_stub(self):
        _, result = _run()
        assert result.error is None
        assert result.corpus_source == "collection"
        assert result.corpus_stats["chunks"] == 3
        assert {c.chunk_id for c in result.retrieved_chunks} <= COLLECTION_CHUNKS
        assert not any(c.chunk_id.startswith("chunk-0") for c in result.retrieved_chunks)

    def test_bm25_retrieval_order_and_cap(self):
        _, result = _run(retrieval_top_k=2)
        # Rank-1 hits of both queries; the off-topic grid paper is cut by the cap.
        assert [c.chunk_id for c in result.retrieved_chunks] == [CAN_GNN_CHUNK, TRANSFER_CHUNK]

    def test_retrieved_chunks_keep_provenance(self):
        _, result = _run()
        chunk = next(c for c in result.retrieved_chunks if c.chunk_id == CAN_GNN_CHUNK)
        assert chunk.paper_id == "s2-can-gnn"
        assert chunk.section == "Abstract"
        assert chunk.text.startswith("Graph Neural Networks for CAN Bus Intrusion Detection")
        assert chunk.metadata["doi"] == "10.1000/can.gnn.2023"
        assert chunk.metadata["providers"] == ["semantic_scholar", "openalex"]
        assert chunk.metadata["corpus"] == "collection"

    def test_detection_prompt_has_only_collected_abstracts(self):
        llm, _ = _run()
        assert [kind for kind, _ in llm.calls] == ["understanding", "detection"]
        prompt = dict(llm.calls)["detection"]
        assert f"[chunk_id: {CAN_GNN_CHUNK}] (paper: s2-can-gnn; section: Abstract)" in prompt
        assert "real-time embedded hardware" in prompt
        assert "chunk-003-limits" not in prompt

    def test_grounding_to_collected_papers(self):
        _, result = _run()
        assert len(result.candidate_gaps) == 2  # stub-chunk and no-abstract gaps dropped
        papers = {p for g in result.candidate_gaps for p in g.source_papers}
        assert papers == {"s2-can-gnn", "oa-transfer"}
        passages = {p for g in result.candidate_gaps for p in g.relevant_passages}
        assert passages <= COLLECTION_CHUNKS

    def test_ranking_prefers_broader_evidence(self):
        _, result = _run()
        top = result.top_k_gaps
        # 0.7 confidence over 2 papers outranks 0.8 confidence over 1 paper.
        assert [g.primary_gap_type for g in top] == [GapType.DATASET, GapType.EVALUATION]
        assert [g.rank for g in top] == [1, 2]
        assert top[0].source_papers == ["oa-transfer", "s2-can-gnn"]
        assert top[0].relevant_passages == [TRANSFER_CHUNK, CAN_GNN_CHUNK]
        assert result.stage_reached == PipelineStage.CANDIDATE_RANKING

    def test_custom_retriever_factory(self):
        seen = {}

        def factory(corpus):
            seen["corpus"] = corpus
            return SparseRetriever(corpus_store=corpus)

        _, result = _run(collection_retriever_factory=factory)
        assert isinstance(seen["corpus"], CollectionCorpusStore)
        assert result.corpus_source == "collection"


class TestNoUsableCollection:
    """
    M2 regression: collection ran but yielded no usable abstracts.  The run
    stops at DOCUMENT_PROCESSING with a clear error; no stub fallback, no
    retrieval, no detection call.
    """

    @pytest.mark.parametrize(
        ("collection", "status"),
        [
            (CollectionResult(query_id="r", status="success"), "success"),
            (CollectionResult(query_id="r", status="failed",
                              provider_errors=[{"provider": "all", "query": "", "error": "x"}]),
             "failed"),
            (CollectionResult(query_id="r", papers=[
                cp for cp in make_collection_result().papers
                if cp.paper.paper_id in ("s2-survey", "s2-blank")
            ]), "success"),
        ],
        ids=["empty_result", "failed_collection", "missing_abstracts"],
    )
    def test_stops_with_clear_error_and_no_stub_fallback(self, collection, status):
        llm = ScriptedFakeLLM(_understanding(), _detection())
        result = _pipeline(llm, _collection_service(collection)).run(Q_MAIN)
        assert result.stage_reached == PipelineStage.DOCUMENT_PROCESSING
        assert result.error.startswith("No usable abstracts in the collected corpus")
        assert f"collection status: {status}" in result.error
        assert result.corpus_source == "collection"
        assert result.retrieved_chunks == []
        assert result.candidate_gaps == [] and result.top_k_gaps == []
        assert [kind for kind, _ in llm.calls] == ["understanding"]  # detection skipped
        # the default _pipeline retriever raises if the stub path is touched

    def test_missing_abstracts_counted_in_error(self):
        collection = CollectionResult(query_id="r", papers=[
            cp for cp in make_collection_result().papers
            if cp.paper.paper_id in ("s2-survey", "s2-blank")
        ])
        result = _pipeline(ScriptedFakeLLM(_understanding(), _detection()),
                           _collection_service(collection)).run(Q_MAIN)
        assert "2 records, 2 unique papers, 2 without abstract" in result.error
        assert result.corpus_stats["skipped_no_abstract"] == 2

    def test_ingestion_error_cause_in_pipeline_error(self):
        """Fix 3: the Stage 2 cause is preserved in PipelineResult.error."""
        service = _collection_service(
            raises=IngestionError("Semantic Scholar 503; OpenAlex timeout")
        )
        result = _pipeline(ScriptedFakeLLM(_understanding(), _detection()), service).run(Q_MAIN)
        assert result.stage_reached == PipelineStage.DOCUMENT_PROCESSING
        assert "1 provider error; first (all): Semantic Scholar 503; OpenAlex timeout" in result.error
        assert result.error.endswith("Candidate gap detection was not run.")

    def test_provider_errors_counted_truncated_and_redacted(self):
        secret = "sk-proj-SECRET1234567890abcd"
        long_tail = "x" * 5000
        collection = CollectionResult(
            query_id="r",
            status="failed",
            provider_errors=[
                {"provider": "semantic_scholar", "query": "q",
                 "error": f"HTTP 401 for key {secret}: {long_tail}"},
                {"provider": "openalex", "query": "q", "error": "timeout"},
            ],
        )
        result = _pipeline(ScriptedFakeLLM(_understanding(), _detection()),
                           _collection_service(collection)).run(Q_MAIN)
        assert "2 provider errors; first (semantic_scholar): HTTP 401 for key [REDACTED]" in result.error
        assert secret not in result.error
        assert "x" * 300 not in result.error  # truncated
        assert len(result.error) < 600
        assert "openalex" not in result.error  # only the first error is quoted

    def test_no_provider_error_summary_when_none(self):
        collection = CollectionResult(query_id="r", status="success")
        result = _pipeline(ScriptedFakeLLM(_understanding(), _detection()),
                           _collection_service(collection)).run(Q_MAIN)
        assert "provider error" not in result.error
        assert result.error == (
            "No usable abstracts in the collected corpus (collection status: success; "
            "0 records, 0 unique papers, 0 without abstract). "
            "Candidate gap detection was not run."
        )

    def test_ingestion_error_does_not_fall_back_to_stub(self):
        llm = ScriptedFakeLLM(_understanding(), _detection())
        service = _collection_service(raises=IngestionError("all providers down"))
        result = _pipeline(llm, service).run(Q_MAIN)
        assert result.collection_result.status == "failed"
        assert result.corpus_source == "collection"
        assert result.stage_reached == PipelineStage.DOCUMENT_PROCESSING
        assert "collection status: failed" in result.error
        assert result.retrieved_chunks == []
        assert [kind for kind, _ in llm.calls] == ["understanding"]


class TestStage3And4Failures:
    """L2 regression: corpus-building / retriever-factory / retrieval failures → FAILED."""

    def _run_with(self, **kwargs):
        llm = ScriptedFakeLLM(_understanding(), _detection())
        result = _pipeline(
            llm, _collection_service(make_collection_result()), **kwargs
        ).run(Q_MAIN)
        return llm, result

    def test_retriever_factory_unexpected_error_wrapped(self):
        def factory(corpus):
            raise ValueError("bad tokenizer config")

        llm, result = self._run_with(collection_retriever_factory=factory)
        assert result.stage_reached == PipelineStage.FAILED
        assert result.error == "Corpus construction failed: ValueError: bad tokenizer config"
        assert result.corpus_source == "collection"
        assert result.corpus_stats["chunks"] == 3  # corpus was built before the factory failed
        assert [kind for kind, _ in llm.calls] == ["understanding"]

    def test_retriever_factory_ewrgv_error_message_kept(self):
        def factory(corpus):
            raise RetrievalError("index unavailable")

        _, result = self._run_with(collection_retriever_factory=factory)
        assert result.stage_reached == PipelineStage.FAILED
        assert result.error == "index unavailable"

    def test_corpus_construction_error_wrapped(self, monkeypatch):
        def boom(cls, result):
            raise KeyError("provenance")

        monkeypatch.setattr(CollectionCorpusStore, "from_collection_result", classmethod(boom))
        llm, result = self._run_with()
        assert result.stage_reached == PipelineStage.FAILED
        assert result.error.startswith("Corpus construction failed: KeyError")
        assert result.retrieved_chunks == []
        assert [kind for kind, _ in llm.calls] == ["understanding"]

    def test_stage3_unexpected_error_logs_traceback(self, caplog):
        def factory(corpus):
            raise ValueError("bad tokenizer config")

        with caplog.at_level("ERROR", logger="app.orchestration.research_pipeline"):
            self._run_with(collection_retriever_factory=factory)
        record = next(r for r in caplog.records if "Stage 3 failed" in r.getMessage())
        assert record.exc_info is not None
        assert record.exc_info[0] is ValueError

    def test_stage4_unexpected_error_on_collection_path(self, caplog):
        """Fix 2: non-EWRGV exceptions from retrieval become a FAILED result."""
        class BrokenRetriever:
            def retrieve(self, query, top_k=20):
                raise ZeroDivisionError("division by zero in scoring")

        with caplog.at_level("ERROR", logger="app.orchestration.research_pipeline"):
            llm, result = self._run_with(collection_retriever_factory=lambda c: BrokenRetriever())
        assert result.stage_reached == PipelineStage.FAILED
        assert result.error == "Retrieval failed: ZeroDivisionError: division by zero in scoring"
        assert result.corpus_source == "collection"
        assert result.retrieved_chunks == []
        assert [kind for kind, _ in llm.calls] == ["understanding"]
        record = next(r for r in caplog.records if "Stage 4 failed" in r.getMessage())
        assert record.exc_info is not None and record.exc_info[0] is ZeroDivisionError

    def test_stage4_unexpected_error_on_injected_retriever_path(self):
        """Fix 2 also applies when no collection ran (injected retriever)."""
        retriever = MagicMock()
        retriever.retrieve.side_effect = KeyError("missing index")
        llm = ScriptedFakeLLM(_understanding(), _detection())
        result = _pipeline(llm, collection_service=None, retriever=retriever).run(Q_MAIN)
        assert result.stage_reached == PipelineStage.FAILED
        assert result.error == "Retrieval failed: KeyError: 'missing index'"
        assert result.corpus_source == "injected_retriever"

    def test_bm25_retrieval_error_on_collection_path(self):
        class FailingRetriever:
            def retrieve(self, query, top_k=20):
                raise RetrievalError("BM25 failed")

        llm, result = self._run_with(collection_retriever_factory=lambda c: FailingRetriever())
        assert result.stage_reached == PipelineStage.FAILED
        assert result.error == "BM25 failed"
        assert result.corpus_source == "collection"
        assert [kind for kind, _ in llm.calls] == ["understanding"]


class TestCrossProviderDuplicatesInPipeline:
    """H1 at pipeline level: one paper from two providers reaches the LLM once."""

    def test_duplicate_paper_sent_once_and_counted_once(self):
        from app.domain.models.collection import CollectedPaper, PaperProvenance
        from app.domain.models.paper import Author, Paper

        def rec(pid, source, doi, abstract):
            return CollectedPaper(
                paper=Paper(paper_id=pid, title="GNN CAN Bus Intrusion Detection",
                            abstract=abstract, doi=doi, publication_year=2023,
                            authors=[Author(name="Arun Kumar")]),
                provenance=PaperProvenance(source=source, provider_id=pid, source_query=Q_MAIN),
            )

        collection = CollectionResult(query_id="r", papers=[
            rec("s2", "semantic_scholar", None, "GNN intrusion detection for CAN bus."),
            rec("oa", "openalex", "10.1/gnn",
                "GNN intrusion detection for CAN bus, evaluated on one vehicle only."),
        ])
        llm = ScriptedFakeLLM(_understanding(), {"candidate_gaps": [{
            "gap_statement": "Single-vehicle evaluation.", "gap_type": "Dataset",
            "supporting_chunk_ids": ["oa:abstract", "s2:abstract"], "initial_confidence": 0.5,
        }]})
        result = _pipeline(llm, _collection_service(collection)).run(Q_MAIN)

        assert [c.chunk_id for c in result.retrieved_chunks] == ["oa:abstract"]
        gap = result.top_k_gaps[0]
        assert gap.source_papers == ["oa"]  # counted once, not twice
        assert gap.metadata["invalid_chunk_ids"] == ["s2:abstract"]
        assert gap.metadata["ranking"]["distinct_papers"] == 1


class TestPipelineResultCompatibility:
    def test_positional_construction_of_original_fields(self):
        """L3 regression: original Phase 2.2 field order is unchanged."""
        from app.orchestration.research_pipeline import PipelineResult

        result = PipelineResult(None, None, [], [], [], PipelineStage.FAILED, "boom")
        assert result.stage_reached == PipelineStage.FAILED
        assert result.error == "boom"
        assert result.retrieved_chunks == []
        assert result.corpus_source is None
        assert result.corpus_stats == {}


class TestFallbackWithoutCollection:
    def test_injected_stub_retriever_used_when_no_collection(self):
        llm = ScriptedFakeLLM(_understanding(), {"candidate_gaps": []})
        result = _pipeline(llm, collection_service=None, retriever=RetrievalService(),
                           retrieval_top_k=3).run("GPT-3 limitations structured prediction")
        assert result.collection_result is None
        assert result.corpus_source == "injected_retriever"
        assert result.corpus_stats == {}
        assert result.retrieved_chunks
        assert all(c.chunk_id.startswith("chunk-0") for c in result.retrieved_chunks)
        assert result.stage_reached == PipelineStage.CANDIDATE_RANKING


class TestEvidenceScoringInPipeline:
    """Stage 6b: optional pre-validation evidence scoring over collected papers."""

    def _detection_with_counter_evidence(self) -> dict:
        payload = _detection()
        # Gap 1 (embedded hardware) is contradicted by the transfer-learning abstract.
        payload["candidate_gaps"][0]["contradicting_chunk_ids"] = [TRANSFER_CHUNK]
        return payload

    def test_scored_candidates_ranked_with_evidence(self):
        from app.domain.enums import CandidateEvidenceStatus
        from app.gap_detection.evidence_scoring import CandidateEvidenceScorer

        llm = ScriptedFakeLLM(_understanding(), self._detection_with_counter_evidence())
        result = _pipeline(llm, _collection_service(make_collection_result()),
                           evidence_scorer=CandidateEvidenceScorer()).run(Q_MAIN)

        assert result.error is None
        assert result.stage_reached == PipelineStage.CANDIDATE_RANKING
        assessments = {g.gap_statement[:20]: g.evidence_assessment for g in result.candidate_gaps}
        assert all(a is not None and a.is_validated is False for a in assessments.values())

        statuses = {g.primary_gap_type: g.evidence_assessment.status for g in result.top_k_gaps}
        assert statuses[GapType.EVALUATION] in (
            CandidateEvidenceStatus.CONTESTED, CandidateEvidenceStatus.CONTRADICTED
        )
        assert statuses[GapType.DATASET] == CandidateEvidenceStatus.SUPPORTED

        dataset_gap = next(g for g in result.top_k_gaps if g.primary_gap_type == GapType.DATASET)
        refs = [(i.chunk_id, i.paper_id) for i in dataset_gap.evidence_assessment.supporting]
        assert refs == [(TRANSFER_CHUNK, "oa-transfer"), (CAN_GNN_CHUNK, "s2-can-gnn")]
        assert dataset_gap.evidence_assessment.supporting[0].publication_year == 2022
        assert dataset_gap.metadata["ranking"]["evidence_score"] == (
            dataset_gap.evidence_assessment.score
        )
        assert result.top_k_gaps[0].primary_gap_type == GapType.DATASET

    def test_without_scorer_behaviour_unchanged(self):
        llm = ScriptedFakeLLM(_understanding(), self._detection_with_counter_evidence())
        result = _pipeline(llm, _collection_service(make_collection_result())).run(Q_MAIN)
        assert all(g.evidence_assessment is None for g in result.candidate_gaps)
        assert all(g.metadata["ranking"]["evidence_score"] is None for g in result.top_k_gaps)
        assert [g.primary_gap_type for g in result.top_k_gaps] == [
            GapType.DATASET, GapType.EVALUATION
        ]

    def test_scorer_failure_returns_failed_result(self):
        class BrokenScorer:
            def score_all(self, gaps, chunks, paper_lookup=None):
                raise ValueError("bad weights at runtime")

        llm = ScriptedFakeLLM(_understanding(), _detection())
        result = _pipeline(llm, _collection_service(make_collection_result()),
                           evidence_scorer=BrokenScorer()).run(Q_MAIN)
        assert result.stage_reached == PipelineStage.FAILED
        assert result.error == "Candidate evidence scoring failed: ValueError: bad weights at runtime"
        assert len(result.candidate_gaps) == 2  # unscored candidates kept for inspection
        assert result.top_k_gaps == []
