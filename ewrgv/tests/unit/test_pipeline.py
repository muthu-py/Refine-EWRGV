"""
tests/unit/test_pipeline.py
-----------------------------
Unit tests for the ResearchPipeline skeleton.

Tests that the pipeline can be constructed and that run() returns a
PipelineResult even when all dependencies are mocked.
"""

from __future__ import annotations

import pytest

from app.domain.enums import PipelineStage
from app.orchestration.research_pipeline import PipelineResult, ResearchPipeline


class _NullObject:
    """Minimal null object satisfying all dependency slots."""

    def __getattr__(self, name):
        def noop(*args, **kwargs):
            return None
        return noop


class TestResearchPipeline:
    def _make_pipeline(self) -> ResearchPipeline:
        null = _NullObject()
        return ResearchPipeline(
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

    def test_run_returns_empty_results_when_unimplemented(self):
        pipeline = self._make_pipeline()
        result = pipeline.run("test query")
        assert result.candidate_gaps == []
        assert result.validation_results == []

    def test_pipeline_result_defaults(self):
        r = PipelineResult()
        assert r.stage_reached == PipelineStage.CREATED
        assert r.error is None
