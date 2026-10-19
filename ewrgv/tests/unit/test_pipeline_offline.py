"""
tests/unit/test_pipeline_offline.py
-------------------------------------
Credential-free end-to-end test of pipeline Stages 1 → 7.

Unlike tests/unit/test_pipeline.py (which mocks the QueryUnderstandingService),
this runs the REAL QueryUnderstandingService, RetrievalService (stub corpus),
GapDetector and CandidateRanker.  Only the LLM is replaced, by a strict
scripted fake that implements exactly the LLMProvider protocol — calling any
other method raises AttributeError, unlike an unspecced MagicMock.

No API key, no network.  Exercises the stub corpus only — NOT evidence that
detection works on collected research papers.
"""

from __future__ import annotations

from app.domain.enums import GapType, PipelineStage
from app.domain.interfaces import LLMProvider
from app.gap_detection.detector import GapDetector
from app.gap_detection.ranking import CandidateRanker
from app.orchestration.research_pipeline import ResearchPipeline
from app.query_understanding.service import QueryUnderstandingService
from app.retrieval.service import RetrievalService
from tests.fixtures.fake_llm import ScriptedFakeLLM

QUESTION = "GPT-3 limitations structured prediction reasoning"


def _understanding_payload() -> dict:
    return {
        "domain": "NLP",
        "problem": "limitations of large language models",
        "task": "structured prediction",
        "methods": ["GPT-3"],
        "concepts": ["structured prediction", "reasoning"],
        "entities": [],
        "datasets": [],
        "metrics": [],
        "constraints": [],
        "research_intent": "gap identification",
        "terminology_variants": [],
        "method_variants": [],
        "expanded_queries": ["repetitive text generation"],
    }


def _detection_payload() -> dict:
    return {
        "candidate_gaps": [
            {
                "gap_statement": "GPT-3 remains weak at structured prediction and precise reasoning.",
                "gap_type": "Methodological",
                "secondary_gap_types": ["Evaluation"],
                "supporting_chunk_ids": ["chunk_id: chunk-003-limits"],
                "initial_confidence": 0.8,
                "rationale": "The Limitations section states this directly.",
                "domain": "NLP",
                "methodology": "GPT-3",
                "key_concepts": ["GPT-3", "structured prediction"],
                "synonyms": ["large language model"],
            },
            {
                "gap_statement": "Text generation by GPT-3 can be repetitive.",
                "gap_type": "Methodological",
                "supporting_chunk_ids": ["chunk-003-limits", "chunk-005-intro"],
                "initial_confidence": 0.6,
            },
            {
                # Cites a corpus chunk that was NOT retrieved → ungrounded, dropped.
                "gap_statement": "Vision Transformers need more data than CNNs.",
                "gap_type": "Dataset",
                "supporting_chunk_ids": ["chunk-004-vit"],
                "initial_confidence": 0.9,
            },
        ]
    }


def _run() -> tuple[ScriptedFakeLLM, object]:
    llm = ScriptedFakeLLM(_understanding_payload(), _detection_payload())
    pipeline = ResearchPipeline(
        query_understanding_service=QueryUnderstandingService(llm=llm),
        literature_collector=None,
        document_parser=None,
        retriever=RetrievalService(),
        reranker=None,
        gap_detector=GapDetector(llm),
        candidate_ranker=CandidateRanker(),
        ewrgv_validator=None,
        top_k_gaps=5,
        retrieval_top_k=3,
    )
    return llm, pipeline.run(QUESTION)


class TestOfflinePipelineStages1To7:
    def test_fake_satisfies_llm_provider_protocol(self):
        assert isinstance(ScriptedFakeLLM({}, {}), LLMProvider)

    def test_reaches_candidate_ranking_without_error(self):
        _, result = _run()
        assert result.error is None
        assert result.stage_reached == PipelineStage.CANDIDATE_RANKING

    def test_exactly_two_llm_calls_in_order(self):
        llm, _ = _run()
        assert [kind for kind, _ in llm.calls] == ["understanding", "detection"]

    def test_query_understanding_drives_retrieval(self):
        _, result = _run()
        assert result.query.understanding.domain == "NLP"
        assert result.query.expanded_queries == ["repetitive text generation"]
        # Original + expanded query merged and capped at retrieval_top_k=3.
        assert [c.chunk_id for c in result.retrieved_chunks] == [
            "chunk-003-limits", "chunk-005-intro", "chunk-002-pretrain",
        ]

    def test_detection_prompt_contains_understanding_and_only_retrieved_chunks(self):
        llm, _ = _run()
        prompt = dict(llm.calls)["detection"]
        assert QUESTION in prompt
        assert "- Domain: NLP" in prompt
        for cid in ("chunk-003-limits", "chunk-005-intro", "chunk-002-pretrain"):
            assert f"[chunk_id: {cid}]" in prompt
        assert "chunk-004-vit" not in prompt

    def test_grounding_and_ranking(self):
        _, result = _run()
        # Ungrounded ViT candidate dropped; decorated ID resolved.
        assert [g.gap_statement for g in result.candidate_gaps] == [
            "GPT-3 remains weak at structured prediction and precise reasoning.",
            "Text generation by GPT-3 can be repetitive.",
        ]
        top = result.top_k_gaps
        assert [g.rank for g in top] == [1, 2]
        first = top[0]
        assert first.primary_gap_type == GapType.METHODOLOGICAL
        assert first.secondary_gap_types == [GapType.EVALUATION]
        assert first.relevant_passages == ["chunk-003-limits"]
        assert first.source_papers == ["paper-003"]
        assert first.structured_representation.key_concepts == ["GPT-3", "structured prediction"]
        assert first.metadata["ranking"]["base_score"] > 0
        second = top[1]
        assert second.source_papers == ["paper-003", "paper-005"]

    def test_validation_stage_not_run(self):
        _, result = _run()
        assert result.validation_results == []
