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

from app.domain.enums import PipelineStage
from app.domain.models.query import ResearchQuery
from app.orchestration.research_pipeline import PipelineResult, ResearchPipeline
from app.query_understanding.service import QueryUnderstandingService


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

    def test_stage_reached_is_query_expansion_after_stage1(self):
        """After a successful Stage 1, stage_reached must be QUERY_EXPANSION."""
        pipeline = self._make_pipeline()
        result = pipeline.run("test query")
        assert result.stage_reached == PipelineStage.QUERY_EXPANSION

    def test_run_returns_empty_gaps_and_validations(self):
        """Stages 2–12 are not yet implemented."""
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
