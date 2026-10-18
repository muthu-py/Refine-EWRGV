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
    4. Hybrid Retrieval (semantic + BM25 + fusion)   ← Phase 3 (wired)
    5. Reranking
    6. Gap Detection                     ← Phase 3 (implemented)
    7. Candidate Ranking → Top-K         ← Phase 3 (implemented)
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

Corpus selection (Stages 3–4)
----------------------------
* When Stage 2 produced a CollectionResult, Stage 3 builds an in-memory
  CollectionCorpusStore from the collected papers' titles + abstracts and
  Stage 4 retrieves from it with BM25 (``collection_retriever_factory``).
  The injected ``retriever`` (stub corpus in development) is NOT used, even
  if the collection is empty or failed — gaps must never come from an
  unrelated corpus.  If the collected corpus has no usable abstracts, the
  run stops at DOCUMENT_PROCESSING with ``error`` explaining why.
  Corpus-construction or retriever-factory failures end the run as FAILED.
* When no collection ran (no LiteratureCollectionService injected), Stage 4
  uses the injected ``retriever`` as before.
``PipelineResult.corpus_source`` records which path was taken.

Limitation: PDF full text is not parsed yet, so collected-paper chunks are
abstract-level only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

from app.core.exceptions import EWRGVError, IngestionError, ProviderError, RetrievalError
from app.core.logging import get_logger
from app.domain.enums import PipelineStage
from app.domain.interfaces import CorpusStore
from app.domain.models.candidate_gap import CandidateGap
from app.domain.models.collection import CollectionResult
from app.domain.models.paper import DocumentChunk
from app.domain.models.query import ResearchQuery
from app.domain.models.retrieval_result import RetrievalResult
from app.domain.models.validation import ValidationResult
from app.gap_detection.llm_call import redact_secrets
from app.ingestion.collectors.service import LiteratureCollectionService
from app.ingestion.corpus.collection_corpus import CollectionCorpusStore
from app.query_understanding.service import QueryUnderstandingService
from app.retrieval.bm25 import SparseRetriever

CORPUS_SOURCE_COLLECTION = "collection"
CORPUS_SOURCE_INJECTED = "injected_retriever"

# Cap on the provider error text copied into PipelineResult.error.
_MAX_PROVIDER_ERROR_CHARS = 200


def _provider_error_summary(collection: CollectionResult) -> str:
    """
    Summarise the collection's provider errors for PipelineResult.error.

    Returns "" when there are none; otherwise the count plus the first error,
    with credentials redacted and the text truncated.
    """
    errors = collection.provider_errors or []
    if not errors:
        return ""
    first = errors[0] if isinstance(errors[0], dict) else {"error": str(errors[0])}
    text = redact_secrets(" ".join(str(first.get("error") or "unknown error").split()))
    if len(text) > _MAX_PROVIDER_ERROR_CHARS:
        text = text[:_MAX_PROVIDER_ERROR_CHARS].rstrip() + " …"
    provider = redact_secrets(str(first.get("provider") or "unknown provider"))
    count = len(errors)
    noun = "error" if count == 1 else "errors"
    return f" {count} provider {noun}; first ({provider}): {text}"


def _default_collection_retriever(corpus: CorpusStore) -> SparseRetriever:
    """BM25 over the collected-papers corpus (the only meaningful retriever until
    real embeddings exist; the stub embeddings are hash-based)."""
    return SparseRetriever(corpus_store=corpus)

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
        Error message if the pipeline failed or could not proceed.
    retrieved_chunks:
        Chunks retrieved in Stage 4 and passed to gap detection.
    corpus_source:
        ``"collection"`` when Stage 4 retrieved from the collected papers,
        ``"injected_retriever"`` when it used the injected retriever.
    corpus_stats:
        CollectionCorpusStore statistics (collection path only).

    Fields added after Phase 2.2 are appended at the end so positional
    construction of the original fields keeps working.
    """

    query: Optional[ResearchQuery] = None
    collection_result: Optional[CollectionResult] = None
    candidate_gaps: list[CandidateGap] = field(default_factory=list)
    top_k_gaps: list[CandidateGap] = field(default_factory=list)
    validation_results: list[ValidationResult] = field(default_factory=list)
    stage_reached: PipelineStage = PipelineStage.CREATED
    error: Optional[str] = None
    retrieved_chunks: list[DocumentChunk] = field(default_factory=list)
    corpus_source: Optional[str] = None
    corpus_stats: dict = field(default_factory=dict)


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
    retrieval_top_k      : int
        Results requested per query in Stage 4, and the cap on the merged
        chunk list passed to gap detection.
    collection_retriever_factory : Callable[[CorpusStore], retriever]
        Builds the Stage 4 retriever over the collected-papers corpus.
        Defaults to BM25 (SparseRetriever).
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
        retrieval_top_k: int = 20,
        collection_retriever_factory: Optional[Callable[[CorpusStore], Any]] = None,
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
        self._retrieval_top_k = retrieval_top_k
        self._collection_retriever_factory = (
            collection_retriever_factory or _default_collection_retriever
        )

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

        # -------------------------------------------------------------- #
        # Stage 3: Corpus construction
        # Collected papers → title + abstract chunks.
        # TODO: PDF parsing + sectioning + chunking (self._parser).
        # -------------------------------------------------------------- #
        corpus: Optional[CollectionCorpusStore] = None
        if result.collection_result is not None:
            result.corpus_source = CORPUS_SOURCE_COLLECTION
            try:
                corpus = CollectionCorpusStore.from_collection_result(result.collection_result)
                result.corpus_stats = dict(corpus.stats)
                retriever = self._collection_retriever_factory(corpus)
            except Exception as exc:  # noqa: BLE001 — wrapped like other stage failures
                error = (
                    exc
                    if isinstance(exc, EWRGVError)
                    else IngestionError(
                        f"Corpus construction failed: {type(exc).__name__}: {exc}"
                    )
                )
                logger.error(
                    "ResearchPipeline: Stage 3 failed (corpus construction)",
                    extra={"error": str(error)},
                    exc_info=True,
                )
                result.error = str(error)
                result.stage_reached = PipelineStage.FAILED
                return result

            result.stage_reached = PipelineStage.DOCUMENT_PROCESSING
            if not corpus.get_all_chunks():
                # Never fall back to the injected (stub) corpus after a
                # collection attempt: gaps would describe unrelated papers.
                result.error = (
                    "No usable abstracts in the collected corpus "
                    f"(collection status: {result.collection_result.status}; "
                    f"{corpus.stats['collected_records']} records, "
                    f"{corpus.stats['unique_papers']} unique papers, "
                    f"{corpus.stats['skipped_no_abstract']} without abstract)."
                    f"{_provider_error_summary(result.collection_result)} "
                    "Candidate gap detection was not run."
                )
                logger.warning(
                    "ResearchPipeline: Stage 3 — collected corpus has no usable "
                    "abstracts; stopping (stub corpus NOT used).",
                    extra={"query_id": enriched_query.query_id, **corpus.stats},
                )
                return result
            logger.info(
                "ResearchPipeline: Stage 3 complete",
                extra={"query_id": enriched_query.query_id, **corpus.stats},
            )
        else:
            retriever = self._retriever
            result.corpus_source = CORPUS_SOURCE_INJECTED
            logger.debug(
                "ResearchPipeline: Stage 3 skipped — no collection result; "
                "using the injected retriever's corpus."
            )

        # -------------------------------------------------------------- #
        # Stage 4: Retrieval over the original + expanded queries
        # -------------------------------------------------------------- #
        try:
            chunks = self._retrieve_chunks(enriched_query, retriever, corpus)
        except EWRGVError as exc:
            logger.error(
                "ResearchPipeline: Stage 4 failed (retrieval)",
                extra={"error": str(exc)},
            )
            result.error = str(exc)
            result.stage_reached = PipelineStage.FAILED
            return result
        except Exception as exc:  # noqa: BLE001 — wrapped like Stage 3 failures
            error = RetrievalError(f"Retrieval failed: {type(exc).__name__}: {exc}")
            logger.error(
                "ResearchPipeline: Stage 4 failed (unexpected retrieval error)",
                extra={"error": str(error)},
                exc_info=True,
            )
            result.error = str(error)
            result.stage_reached = PipelineStage.FAILED
            return result

        result.retrieved_chunks = chunks
        result.stage_reached = PipelineStage.RETRIEVAL
        logger.info(
            "ResearchPipeline: Stage 4 complete",
            extra={"query_id": enriched_query.query_id, "chunks": len(chunks)},
        )

        # TODO Stage 5: reranking (self._reranker)

        # -------------------------------------------------------------- #
        # Stage 6: Candidate gap detection
        # -------------------------------------------------------------- #
        try:
            candidates = self._detector.detect(enriched_query, chunks) or []
        except EWRGVError as exc:
            logger.error(
                "ResearchPipeline: Stage 6 failed (gap detection)",
                extra={"error": str(exc)},
            )
            result.error = str(exc)
            result.stage_reached = PipelineStage.FAILED
            return result

        result.candidate_gaps = list(candidates)
        result.stage_reached = PipelineStage.GAP_DETECTION
        logger.info(
            "ResearchPipeline: Stage 6 complete",
            extra={"query_id": enriched_query.query_id, "candidates": len(candidates)},
        )

        # -------------------------------------------------------------- #
        # Stage 7: Candidate ranking → Top-K
        # -------------------------------------------------------------- #
        result.top_k_gaps = list(
            self._ranker.rank(result.candidate_gaps, top_k=self._top_k_gaps) or []
        )
        result.stage_reached = PipelineStage.CANDIDATE_RANKING
        logger.info(
            "ResearchPipeline: Stage 7 complete",
            extra={"query_id": enriched_query.query_id, "top_k": len(result.top_k_gaps)},
        )

        # TODO Stage 8: EWRGV validation (self._validator.validate_batch)
        # TODO Stage 9–12: handled inside EWRGVValidator

        logger.info(
            "ResearchPipeline.run: Stages 1-4, 6-7 complete; "
            "Stages 5 and 8-12 not yet implemented.",
        )
        return result

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _retrieve_chunks(
        self,
        query: ResearchQuery,
        retriever: Any,
        corpus: Optional[CorpusStore] = None,
    ) -> list[DocumentChunk]:
        """
        Retrieve for the original query and each expanded query, then merge.

        Merge order: best (lowest) rank across queries first, ties broken by
        the earlier query, then the earlier position.  Duplicate chunk IDs
        are collapsed.  The merged list is capped at ``retrieval_top_k``.

        When ``corpus`` is given, each retrieved chunk is restored from the
        corpus by chunk_id so its provenance metadata survives (some
        retrievers, e.g. SparseRetriever.retrieve, return chunks without it).
        """
        queries: list[str] = []
        for q in [query.query, *query.expanded_queries]:
            if q and q.strip() and q not in queries:
                queries.append(q)

        corpus_chunks = (
            {c.chunk_id: c for c in corpus.get_all_chunks()} if corpus is not None else {}
        )

        best: dict[str, tuple[int, int, DocumentChunk]] = {}
        for query_index, q in enumerate(queries):
            items = retriever.retrieve(q, top_k=self._retrieval_top_k) or []
            for position, item in enumerate(items):
                chunk = _with_corpus_provenance(_to_document_chunk(item), corpus_chunks)
                key = (position + 1, query_index)
                current = best.get(chunk.chunk_id)
                if current is None or key < current[:2]:
                    best[chunk.chunk_id] = (*key, chunk)

        merged = sorted(best.values(), key=lambda entry: (entry[0], entry[1]))
        return [chunk for _, _, chunk in merged[: self._retrieval_top_k]]


def _with_corpus_provenance(
    chunk: DocumentChunk, corpus_chunks: dict[str, DocumentChunk]
) -> DocumentChunk:
    """Restore a retrieved chunk's corpus text and metadata (retrieval keys win)."""
    original = corpus_chunks.get(chunk.chunk_id)
    if original is None:
        return chunk
    return original.model_copy(
        update={"metadata": {**original.metadata, **chunk.metadata}}
    )


def _to_document_chunk(item: DocumentChunk | RetrievalResult) -> DocumentChunk:
    """
    Normalise a retriever output item to a DocumentChunk.

    The domain Retriever protocol returns DocumentChunk, while HybridRetriever
    and RetrievalService return RetrievalResult.  Retrieval scores are kept in
    chunk metadata for provenance.
    """
    if isinstance(item, DocumentChunk):
        return item
    if isinstance(item, RetrievalResult):
        metadata = dict(item.metadata)
        metadata.update(
            {
                "retrieval_score": item.score,
                "retrieval_rank": item.rank,
                "retrieval_method": item.retrieval_method,
            }
        )
        return DocumentChunk(
            chunk_id=item.chunk_id,
            paper_id=item.paper_id,
            section=item.section,
            text=item.text,
            metadata=metadata,
        )
    raise TypeError(f"Unsupported retrieval result type: {type(item).__name__}")
