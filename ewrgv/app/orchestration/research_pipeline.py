"""
app/orchestration/research_pipeline.py
----------------------------------------
ResearchPipeline – end-to-end research workflow orchestrator.

This class is the top-level coordinator of the EWRGV system.
It composes all major modules via dependency injection and drives the
pipeline from research question to final validated results.

Pipeline stages
---------------
    1. Query Understanding & Expansion   ← Phase 2.1 (implemented)
    2. Literature Collection             ← Phase 2.2 (implemented)
    3. Document Processing (parse → section → chunk)
    4. Hybrid Retrieval (semantic + BM25 + fusion)
    5. Reranking
    6. Gap Detection
    7. Candidate Ranking → Top-K
    8. EWRGV Validation (the research contribution)
    9. Evidence Aggregation
   10. Confidence Assessment
   11. Classification
   12. Explanation

Dependency injection
--------------------
All dependencies are injected at construction time.  The pipeline itself
has no direct imports of infrastructure classes.  This keeps the pipeline
testable and ensures infrastructure is swappable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from app.core.exceptions import EWRGVError, IngestionError, ProviderError
from app.core.logging import get_logger
from app.domain.enums import PipelineStage
from app.domain.models.candidate_gap import CandidateGap
from app.domain.models.collection import CollectionResult
from app.domain.models.query import ResearchQuery
from app.domain.models.validation import ValidationResult
from app.ingestion.collectors.service import LiteratureCollectionService
from app.query_understanding.service import QueryUnderstandingService

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """
    Aggregated result from a complete pipeline run.

    Attributes
    ----------
    query:
        The (expanded) research query.
    collection_result:
        Raw literature collected in Phase 2.2 (before deduplication).
    candidate_gaps:
        All candidate gaps detected before EWRGV validation.
    top_k_gaps:
        The ranked top-k candidates passed to EWRGV.
    validation_results:
        One ValidationResult per top-k gap.
    stage_reached:
        The last pipeline stage completed before stopping or failing.
    error:
        Error message if the pipeline failed.
    """

    query: Optional[ResearchQuery] = None
    collection_result: Optional[CollectionResult] = None
    candidate_gaps: list[CandidateGap] = field(default_factory=list)
    top_k_gaps: list[CandidateGap] = field(default_factory=list)
    validation_results: list[ValidationResult] = field(default_factory=list)
    stage_reached: PipelineStage = PipelineStage.CREATED
    error: Optional[str] = None


class ResearchPipeline:
    """
    Orchestrates the full EWRGV research pipeline.

    Constructor parameters (all injected)
    --------------------------------------
    query_understanding_service : QueryUnderstandingService
        Phase 2.1 — query decomposition and expansion.
    literature_collection_service : LiteratureCollectionService
        Phase 2.2 — automatic literature collection.
    document_parser      : DocumentParser
    retriever            : Retriever
    reranker             : Reranker
    gap_detector         : GapDetector
    candidate_ranker     : CandidateRanker
    ewrgv_validator      : EWRGVValidator
    top_k_gaps           : int
    """

    def __init__(
        self,
        query_understanding_service: QueryUnderstandingService,
        literature_collector: object,
        document_parser: object,
        retriever: object,
        reranker: object,
        gap_detector: object,
        candidate_ranker: object,
        ewrgv_validator: object,
        top_k_gaps: int = 5,
        literature_collection_service: Optional[LiteratureCollectionService] = None,
    ) -> None:
        self._qu_service = query_understanding_service
        # Phase 2.2: prefer the high-level service if provided; fall back
        # to the raw collector passed for backward-compat with existing tests.
        self._collection_service = literature_collection_service
        self._collector = literature_collector
        self._parser = document_parser
        self._retriever = retriever
        self._reranker = reranker
        self._detector = gap_detector
        self._ranker = candidate_ranker
        self._validator = ewrgv_validator
        self._top_k_gaps = top_k_gaps

    def run(self, query_text: str) -> PipelineResult:
        """
        Execute the full pipeline for a research question.

        Returns
        -------
        PipelineResult
        """
        logger.info("ResearchPipeline.run called", extra={"query": query_text[:80]})
        result = PipelineResult()

        # -------------------------------------------------------------- #
        # Stage 1: Query Understanding & Expansion (Phase 2.1)
        # -------------------------------------------------------------- #
        raw_query = ResearchQuery(query=query_text)
        try:
            enriched_query = self._qu_service.run(raw_query)
        except ProviderError as exc:
            logger.error(
                "ResearchPipeline: Stage 1 failed (query understanding)",
                extra={"error": str(exc)},
            )
            result.error = str(exc)
            result.stage_reached = PipelineStage.FAILED
            return result

        result.query = enriched_query
        result.stage_reached = PipelineStage.QUERY_EXPANSION
        logger.info(
            "ResearchPipeline: Stage 1 complete",
            extra={
                "query_id": enriched_query.query_id,
                "expanded_count": len(enriched_query.expanded_queries),
            },
        )

        # -------------------------------------------------------------- #
        # Stage 2: Literature Collection (Phase 2.2)
        # -------------------------------------------------------------- #
        if self._collection_service is not None:
            try:
                collection_result = self._collection_service.collect(enriched_query)
                result.collection_result = collection_result
                result.stage_reached = PipelineStage.LITERATURE_COLLECTION

                if collection_result.status == "failed":
                    logger.warning(
                        "ResearchPipeline: Stage 2 — all providers failed; "
                        "continuing with empty corpus.",
                        extra={"query_id": enriched_query.query_id},
                    )
                else:
                    logger.info(
                        "ResearchPipeline: Stage 2 complete",
                        extra={
                            "query_id": enriched_query.query_id,
                            "total_collected": collection_result.total_collected,
                            "status": collection_result.status,
                        },
                    )
            except IngestionError as exc:
                logger.error(
                    "ResearchPipeline: Stage 2 failed (literature collection)",
                    extra={"error": str(exc)},
                )
                # Non-fatal: record error but don't abort the pipeline
                result.collection_result = CollectionResult(
                    query_id=enriched_query.query_id,
                    status="failed",
                    provider_errors=[{"provider": "all", "query": "", "error": str(exc)}],
                )
                result.stage_reached = PipelineStage.LITERATURE_COLLECTION
        else:
            logger.debug(
                "ResearchPipeline: Stage 2 skipped — no LiteratureCollectionService injected."
            )

        # TODO Stage 3: document parsing + chunking (self._parser)
        # TODO Stage 4: hybrid retrieval (self._retriever)
        # TODO Stage 5: reranking (self._reranker)
        # TODO Stage 6: gap detection (self._detector)
        # TODO Stage 7: candidate ranking + top-k (self._ranker)
        # TODO Stage 8: EWRGV validation (self._validator.validate_batch)
        # TODO Stage 9–12: handled inside EWRGVValidator

        logger.info(
            "ResearchPipeline.run: Stages 1-2 complete; "
            "remaining stages not yet implemented.",
        )
        return result
