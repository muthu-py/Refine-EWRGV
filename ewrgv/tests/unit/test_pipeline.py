"""
tests/unit/test_pipeline.py
-----------------------------
Unit tests for the ResearchPipeline.

Tests that the pipeline can be constructed and that run() returns a
PipelineResult even when all dependencies are mocked.

Phase 2.1 update: QueryUnderstandingService is now an explicit
injected dependency.  The pipeline test mocks it alongside the others.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.exceptions import GapDetectionError, RetrievalError
from app.domain.enums import PipelineStage
from app.domain.models.candidate_gap import CandidateGap
from app.domain.models.paper_chunk import PaperChunk
from app.domain.models.query import ResearchQuery
from app.domain.models.retrieval_result import RetrievalResult
from app.gap_detection.detector import GapDetector
from app.gap_detection.ranking import CandidateRanker
from app.orchestration.research_pipeline import PipelineResult, ResearchPipeline
from app.query_understanding.service import QueryUnderstandingService
from app.retrieval.service import RetrievalService
from tests.fixtures.sample_corpus import FakeEmbeddingProvider, make_sample_corpus
from tests.fixtures.sample_data import make_chunk


def _hybrid_factory(corpus):
    """Dense + BM25 hybrid over the run corpus with a deterministic embedder."""
    return RetrievalService(corpus, FakeEmbeddingProvider())


class _NullObject:
    """Minimal null object satisfying all dependency slots."""

    def __getattr__(self, name):
        def noop(*args, **kwargs):
            return None
        return noop


def _make_qu_service_mock(return_query: ResearchQuery = None) -> MagicMock:
    """Return a mocked QueryUnderstandingService."""
    mock = MagicMock(spec=QueryUnderstandingService)
    if return_query is not None:
        mock.run.return_value = return_query
    else:
        # Default: return the input query unchanged
        mock.run.side_effect = lambda q: q
    return mock


class TestResearchPipeline:
    def _make_pipeline(
        self,
        qu_service=None,
    ) -> ResearchPipeline:
        null = _NullObject()
        return ResearchPipeline(
            query_understanding_service=qu_service or _make_qu_service_mock(),
            literature_collector=null,
            document_parser=null,
            retriever=null,
            reranker=null,
            gap_detector=null,
            candidate_ranker=null,
            ewrgv_validator=null,
        )

    def test_pipeline_instantiation(self):
        pipeline = self._make_pipeline()
        assert isinstance(pipeline, ResearchPipeline)

    def test_run_returns_pipeline_result(self):
        pipeline = self._make_pipeline()
        result = pipeline.run("What are the gaps in RAG systems?")
        assert isinstance(result, PipelineResult)

    def test_run_returns_query_in_result(self):
        """After Stage 1, result.query should be set."""
        pipeline = self._make_pipeline()
        result = pipeline.run("What are the gaps in RAG systems?")
        assert result.query is not None
        assert isinstance(result.query, ResearchQuery)

    def test_stage_reached_with_null_dependencies(self):
        """
        With null dependencies (no chunks, no gaps), the pipeline still runs
        through Stage 7 and stops at CANDIDATE_RANKING (Stage 8+ not implemented).
        """
        pipeline = self._make_pipeline()
        result = pipeline.run("test query")
        assert result.stage_reached == PipelineStage.CANDIDATE_RANKING

    def test_run_returns_empty_gaps_and_validations(self):
        """Null dependencies yield no gaps; Stage 8+ (validation) is not implemented."""
        pipeline = self._make_pipeline()
        result = pipeline.run("test query")
        assert result.candidate_gaps == []
        assert result.validation_results == []

    def test_pipeline_result_defaults(self):
        r = PipelineResult()
        assert r.stage_reached == PipelineStage.CREATED
        assert r.error is None

    def test_provider_failure_in_stage1_sets_error(self):
        """If the QU service raises ProviderError, the pipeline should fail gracefully."""
        from app.core.exceptions import ProviderError

        failing_service = MagicMock(spec=QueryUnderstandingService)
        failing_service.run.side_effect = ProviderError("LLM timeout")

        pipeline = self._make_pipeline(qu_service=failing_service)
        result = pipeline.run("What are the gaps in RAG systems?")

        assert result.stage_reached == PipelineStage.FAILED
        assert result.error is not None
        assert result.query is None


# ======================================================================
# Stages 4, 6, 7 — retrieval → candidate gap detection → ranking
# ======================================================================


def _result(chunk_id: str, paper_id: str = "p1", rank: int = 1) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        paper_id=paper_id,
        section="Limitations",
        text=f"text of {chunk_id}",
        score=0.5,
        rank=rank,
        retrieval_method="hybrid",
    )


class TestPipelineGapStages:
    def _make_pipeline(self, retriever, detector, ranker=None, query=None, **kwargs):
        null = _NullObject()
        return ResearchPipeline(
            query_understanding_service=_make_qu_service_mock(query),
            literature_collector=null,
            document_parser=null,
            retriever=retriever,
            reranker=null,
            gap_detector=detector,
            candidate_ranker=ranker or CandidateRanker(),
            ewrgv_validator=null,
            **kwargs,
        )

    def test_retrieval_results_converted_to_chunks(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = [_result("c1"), _result("c2", rank=2)]
        detector = MagicMock()
        detector.detect.return_value = []

        result = self._make_pipeline(retriever, detector).run("q")

        chunks = detector.detect.call_args.args[1]
        assert [c.chunk_id for c in chunks] == ["c1", "c2"]
        assert all(isinstance(c, PaperChunk) for c in chunks)
        assert chunks[0].section_title == "Limitations"
        assert result.retrieved_chunks == chunks
        assert result.retrieval_provenance["c1"]["retrieval_method"] == "hybrid"
        assert result.retrieval_provenance["c2"]["retrieval_rank"] == 2

    def test_retrieves_for_original_and_expanded_queries_and_dedupes(self):
        query = ResearchQuery(query="orig", expanded_queries=["exp1", "orig", "exp2"])
        retriever = MagicMock()
        retriever.retrieve.side_effect = lambda q, top_k: {
            "orig": [_result("c1"), _result("c2", rank=2)],
            "exp1": [_result("c3"), _result("c1", rank=2)],
            "exp2": [],
        }[q]
        detector = MagicMock()
        detector.detect.return_value = []

        self._make_pipeline(retriever, detector, query=query).run("orig")

        called = [c.args[0] for c in retriever.retrieve.call_args_list]
        assert called == ["orig", "exp1", "exp2"]
        chunks = detector.detect.call_args.args[1]
        # rank-1 hits first (orig before exp1), then rank-2 hits; c1 not duplicated
        assert [c.chunk_id for c in chunks] == ["c1", "c3", "c2"]

    def test_retrieval_top_k_caps_merged_chunks(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = [_result(f"c{i}", rank=i) for i in range(1, 6)]
        detector = MagicMock()
        detector.detect.return_value = []

        self._make_pipeline(retriever, detector, retrieval_top_k=3).run("q")

        assert retriever.retrieve.call_args.kwargs["top_k"] == 3
        assert len(detector.detect.call_args.args[1]) == 3

    def test_gaps_detected_and_ranked(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = [_result("c1")]
        gaps = [
            CandidateGap(gap_statement=f"gap number {i} about topic {i}",
                         initial_confidence=i / 10,
                         source_papers=["p1"], relevant_passages=["c1"])
            for i in range(1, 8)
        ]
        detector = MagicMock()
        detector.detect.return_value = gaps

        result = self._make_pipeline(retriever, detector, top_k_gaps=3).run("q")

        assert result.stage_reached == PipelineStage.CANDIDATE_RANKING
        assert len(result.candidate_gaps) == 7
        assert [g.rank for g in result.top_k_gaps] == [1, 2, 3]
        assert result.top_k_gaps[0].initial_confidence == 0.7
        assert result.validation_results == []

    def test_gap_detection_error_fails_gracefully(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = [_result("c1")]
        detector = MagicMock()
        detector.detect.side_effect = GapDetectionError("LLM down")

        result = self._make_pipeline(retriever, detector).run("q")

        assert result.stage_reached == PipelineStage.FAILED
        assert "LLM down" in result.error
        assert result.query is not None
        assert len(result.retrieved_chunks) == 1
        assert result.candidate_gaps == []

    def test_retrieval_error_fails_gracefully(self):
        retriever = MagicMock()
        retriever.retrieve.side_effect = RetrievalError("index broken")
        detector = MagicMock()

        result = self._make_pipeline(retriever, detector).run("q")

        assert result.stage_reached == PipelineStage.FAILED
        assert "index broken" in result.error
        detector.detect.assert_not_called()

    def test_end_to_end_on_run_corpus_with_mocked_llm(self):
        """
        Real hybrid retrieval over a RunCorpus of PaperChunks + real
        GapDetector/CandidateRanker, with only the LLM mocked.
        """
        llm = MagicMock()
        llm.complete_structured.return_value = {
            "candidate_gaps": [
                {
                    "gap_statement": "Large language models remain weak at structured prediction.",
                    "gap_type": "Methodological",
                    "supporting_chunk_ids": ["chunk-003-limits", "chunk-not-retrieved"],
                    "initial_confidence": 0.8,
                    "key_concepts": ["GPT-3", "structured prediction"],
                }
            ]
        }
        pipeline = self._make_pipeline(
            retriever=None,
            detector=GapDetector(llm),
            retriever_factory=_hybrid_factory,
        )

        result = pipeline.run("What are the limitations of GPT-3?", corpus=make_sample_corpus())

        assert result.stage_reached == PipelineStage.CANDIDATE_RANKING
        assert result.corpus_source == "run_corpus"
        assert "chunk-003-limits" in {c.chunk_id for c in result.retrieved_chunks}
        assert len(result.top_k_gaps) == 1
        gap = result.top_k_gaps[0]
        assert gap.relevant_passages == ["chunk-003-limits"]
        assert gap.source_papers == ["paper-003"]
        assert gap.metadata["invalid_chunk_ids"] == ["chunk-not-retrieved"]
        assert gap.rank == 1


class TestPipelineGapStagesCoverage:
    """M4 regression coverage for Stage 4 conversion and real retrieval."""

    def _make_pipeline(self, retriever, detector, query=None, **kwargs):
        null = _NullObject()
        return ResearchPipeline(
            query_understanding_service=_make_qu_service_mock(query),
            literature_collector=null,
            document_parser=null,
            retriever=retriever,
            reranker=null,
            gap_detector=detector,
            candidate_ranker=CandidateRanker(),
            ewrgv_validator=null,
            **kwargs,
        )

    def test_plain_paper_chunks_pass_through_unchanged(self):
        """Retrievers satisfying the domain Retriever protocol return PaperChunk."""
        original = [
            make_chunk(chunk_id="d1", paper_id="p1", text="t1"),
            make_chunk(chunk_id="d2", paper_id="p2", text="t2"),
        ]
        retriever = MagicMock()
        retriever.retrieve.return_value = original
        detector = MagicMock()
        detector.detect.return_value = []

        result = self._make_pipeline(retriever, detector).run("q")

        passed = detector.detect.call_args.args[1]
        assert [c.chunk_id for c in passed] == ["d1", "d2"]
        assert passed[0] is original[0]  # no copy, no conversion
        assert result.retrieved_chunks == original
        # PaperChunks carry no score; only the query that found them is recorded.
        assert result.retrieval_provenance == {"d1": {"query": "q"}, "d2": {"query": "q"}}

    def test_mixed_result_types_merged(self):
        retriever = MagicMock()
        retriever.retrieve.return_value = [
            make_chunk(chunk_id="d1", paper_id="p1", text="t1"),
            _result("r1", rank=2),
        ]
        detector = MagicMock()
        detector.detect.return_value = []

        result = self._make_pipeline(retriever, detector).run("q")

        passed = detector.detect.call_args.args[1]
        assert [type(c) for c in passed] == [PaperChunk, PaperChunk]
        assert result.retrieval_provenance["r1"]["retrieval_rank"] == 2

    def test_real_dense_retriever_returning_paper_chunks(self):
        from app.retrieval.semantic import DenseRetriever

        dense = DenseRetriever(
            embedding_provider=FakeEmbeddingProvider(), corpus_store=make_sample_corpus()
        )
        detector = MagicMock()
        detector.detect.return_value = []

        result = self._make_pipeline(dense, detector, retrieval_top_k=4).run("attention")

        assert result.stage_reached == PipelineStage.CANDIDATE_RANKING
        assert len(result.retrieved_chunks) == 4
        assert all(isinstance(c, PaperChunk) for c in result.retrieved_chunks)

    def test_end_to_end_run_corpus_retrieval_with_small_top_k(self):
        """
        Real hybrid retrieval over the sample RunCorpus with retrieval_top_k=3,
        so only part of the 13-chunk corpus reaches the detector.  Grounding
        must be against the *retrieved* chunks: citations of chunks that exist
        in the corpus but were not retrieved are rejected.
        """
        query = ResearchQuery(
            query="GPT-3 limitations structured prediction reasoning",
            expanded_queries=["repetitive text generation"],
        )
        llm = MagicMock()
        llm.complete_structured.return_value = {
            "candidate_gaps": [
                {
                    "gap_statement": "GPT-3 remains weak at structured prediction tasks.",
                    "gap_type": "Methodological",
                    # chunk-004-vit is in the corpus but not retrieved
                    "supporting_chunk_ids": ["[chunk_id: chunk-003-limits]", "chunk-004-vit"],
                    "initial_confidence": 0.8,
                },
                {
                    "gap_statement": "RAG lacks evaluation on low-resource domains.",
                    "gap_type": "Evaluation",
                    "supporting_chunk_ids": ["chunk-005-rag"],  # in corpus, not retrieved
                    "initial_confidence": 0.9,
                },
            ]
        }
        pipeline = self._make_pipeline(
            None, GapDetector(llm), query=query, retrieval_top_k=3,
            retriever_factory=_hybrid_factory,
        )

        result = pipeline.run(query.query, corpus=make_sample_corpus())

        # rank-1 hits of both queries first, then the best rank-2 hit; capped at 3
        assert [c.chunk_id for c in result.retrieved_chunks] == [
            "chunk-003-limits", "chunk-005-intro", "chunk-002-pretrain",
        ]
        prompt = llm.complete_structured.call_args.args[0]
        assert "[chunk_id: chunk-003-limits]" in prompt
        assert "chunk-004-vit" not in prompt and "chunk-005-rag" not in prompt

        assert result.stage_reached == PipelineStage.CANDIDATE_RANKING
        assert [g.gap_statement for g in result.candidate_gaps] == [
            "GPT-3 remains weak at structured prediction tasks."
        ]
        gap = result.top_k_gaps[0]
        assert gap.rank == 1
        assert gap.relevant_passages == ["chunk-003-limits"]
        assert gap.source_papers == ["paper-003"]
        assert gap.metadata["invalid_chunk_ids"] == ["chunk-004-vit"]

    def test_provider_failure_cause_reaches_pipeline_result(self):
        """M3 regression through the real detector, not an injected GapDetectionError."""
        from app.core.exceptions import ProviderError

        secret = "sk-proj-SECRETVALUE123456"
        llm = MagicMock()
        llm.complete_structured.side_effect = ProviderError(
            f"Error code: 429 - Rate limit exceeded (key {secret})"
        )
        retriever = MagicMock()
        retriever.retrieve.return_value = [_result("c1")]

        result = self._make_pipeline(retriever, GapDetector(llm)).run("q")

        assert result.stage_reached == PipelineStage.FAILED
        assert "429" in result.error
        assert "Rate limit exceeded" in result.error
        assert secret not in result.error
