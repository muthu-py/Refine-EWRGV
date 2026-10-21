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
The retrieval corpus is the run's RunCorpus (app.ingestion.corpus): the
full-text PaperChunks produced by /acquire → /parse → /sections → /chunks,
with the embeddings and BM25 index from /represent.  Those phases run as
background jobs, so the corpus is supplied to the pipeline rather than
built inside it:

* ``run(query, corpus=...)`` or ``run_on_corpus(query, corpus)`` — Stage 4
  retrieves from that corpus with ``retriever_factory(corpus)`` (default:
  BM25 via RetrievalService).  An empty corpus stops the run at
  DOCUMENT_PROCESSING with ``error`` explaining why.
* A ``corpus_loader`` (``run_id -> CorpusStore``) may be injected instead;
  it is called after Stage 2 with the collection's run ID.
* If a collection ran but no corpus is available, the run stops after
  Stage 2: the collected papers must be processed first.  The injected
  ``retriever`` is NOT used then — gaps must never come from an unrelated
  corpus.
* When no collection ran and no corpus is given, Stage 4 uses the injected
  ``retriever``.
``PipelineResult.corpus_source`` records which path was taken.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Optional

from app.core.config import settings
from app.core.exceptions import (
    EWRGVError,
    GapDetectionError,
    IngestionError,
    ProviderError,
    RetrievalError,
)
from app.core.logging import get_logger
from app.domain.enums import PipelineStage, SectionType
from app.domain.interfaces import CorpusStore
from app.domain.models.candidate_gap import CandidateGap
from app.domain.models.collection import CollectionResult
from app.domain.models.paper_chunk import PaperChunk
from app.domain.models.query import ResearchQuery
from app.domain.models.retrieval_result import RetrievalResult
from app.domain.models.validation import ValidationResult
from app.gap_detection.evidence_scoring import CandidateEvidenceScorer
from app.gap_detection.llm_call import redact_secrets
from app.ingestion.collectors.service import LiteratureCollectionService
from app.query_understanding.service import QueryUnderstandingService
from app.retrieval.service import RetrievalService
from app.retrieval.reranking import CrossEncoderReranker, get_cross_encoder_reranker

CORPUS_SOURCE_RUN = "run_corpus"
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


def _default_corpus_retriever(corpus: CorpusStore) -> RetrievalService:
    """BM25 over the run corpus (uses the run's persisted index when loaded).

    Returns scored RetrievalResults so Stage 4 can record provenance.  Inject
    a factory passing an embedding provider to RetrievalService to make it
    hybrid (dense + BM25)."""
    return RetrievalService(corpus)

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
        PaperChunks retrieved in Stage 4 and passed to gap detection.
    corpus_source:
        ``"run_corpus"`` when Stage 4 retrieved from the run's corpus,
        ``"injected_retriever"`` when it used the injected retriever.
    corpus_stats:
        RunCorpus statistics (run-corpus path only).
    retrieval_provenance:
        ``chunk_id → {retrieval_score, retrieval_rank, retrieval_method,
        query}`` for each retrieved chunk, when the retriever reports scores.

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
    retrieved_chunks: list[PaperChunk] = field(default_factory=list)
    corpus_source: Optional[str] = None
    corpus_stats: dict = field(default_factory=dict)
    retrieval_provenance: dict[str, dict] = field(default_factory=dict)


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
    retriever_factory : Callable[[CorpusStore], retriever]
        Builds the Stage 4 retriever over the run corpus.
        Defaults to BM25 (RetrievalService without an embedding provider).
    corpus_loader : Callable[[str], CorpusStore], optional
        Returns the processed corpus for a run ID (e.g. a RunCorpus loaded
        ahead of time).  Called after Stage 2 when no corpus is passed in.
    evidence_scorer : CandidateEvidenceScorer, optional
        If given, each detected candidate is given a pre-validation
        evidence-weighted assessment (Stage 6b) before ranking, and the
        ranker blends that score in.  ``None`` (default) keeps the previous
        behaviour.  This is not EWRGV validation.
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
        retriever_factory: Optional[Callable[[CorpusStore], Any]] = None,
        evidence_scorer: Optional[CandidateEvidenceScorer] = None,
        corpus_loader: Optional[Callable[[str], Optional[CorpusStore]]] = None,
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
        self._retriever_factory = retriever_factory or _default_corpus_retriever
        self._evidence_scorer = evidence_scorer
        self._corpus_loader = corpus_loader

    def run(self, query_text: str, corpus: Optional[CorpusStore] = None) -> PipelineResult:
        """
        Execute the full pipeline for a research question.

        Parameters
        ----------
        query_text:
            The research question.
        corpus:
            The processed corpus to detect gaps in (normally the run's
            RunCorpus).  See "Corpus selection" in the module docstring.

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
        # Stage 3: Corpus selection (the run's processed PaperChunks)
        # -------------------------------------------------------------- #
        if corpus is None and self._corpus_loader is not None and result.collection_result is not None:
            run_id = result.collection_result.query_id
            try:
                corpus = self._corpus_loader(run_id)
            except Exception as exc:  # noqa: BLE001 — wrapped like other stage failures
                error = (
                    exc
                    if isinstance(exc, EWRGVError)
                    else IngestionError(f"Corpus loading failed: {type(exc).__name__}: {exc}")
                )
                logger.error(
                    "ResearchPipeline: Stage 3 failed (corpus loading)",
                    extra={"error": str(error), "run_id": run_id},
                    exc_info=True,
                )
                result.error = str(error)
                result.stage_reached = PipelineStage.FAILED
                return result

        if corpus is None and result.collection_result is not None:
            # Never fall back to the injected retriever after a collection:
            # gaps would describe papers that are not in this run.
            result.error = (
                "No processed corpus for this run: the collected papers must be "
                "acquired, parsed, sectioned, chunked and represented "
                "(/acquire → /parse → /sections → /chunks → /represent) before "
                "candidate gap detection."
                f"{_provider_error_summary(result.collection_result)}"
            )
            logger.warning(
                "ResearchPipeline: Stage 3 — no corpus for the collected run; stopping.",
                extra={"query_id": enriched_query.query_id},
            )
            return result

        return self._detect_from_corpus(result, enriched_query, corpus)

    def run_on_corpus(self, query: ResearchQuery, corpus: CorpusStore) -> PipelineResult:
        """
        Run Stages 3–7 on an already processed corpus.

        ``query`` should be the enriched query from Stage 1 (its expanded
        queries drive Stage 4).  Query understanding and collection are not
        repeated.
        """
        result = PipelineResult(query=query, stage_reached=PipelineStage.LITERATURE_COLLECTION)
        return self._detect_from_corpus(result, query, corpus)

    def _detect_from_corpus(
        self,
        result: PipelineResult,
        enriched_query: ResearchQuery,
        corpus: Optional[CorpusStore],
    ) -> PipelineResult:
        """Stages 3 (corpus checks) to 7 (ranking)."""
        if corpus is not None:
            result.corpus_source = CORPUS_SOURCE_RUN
            stats = getattr(corpus, "stats", None)
            result.corpus_stats = dict(stats) if isinstance(stats, dict) else {}
            try:
                retriever = self._retriever_factory(corpus)
            except Exception as exc:  # noqa: BLE001 — wrapped like other stage failures
                error = (
                    exc
                    if isinstance(exc, EWRGVError)
                    else RetrievalError(
                        f"Retriever construction failed: {type(exc).__name__}: {exc}"
                    )
                )
                logger.error(
                    "ResearchPipeline: Stage 3 failed (retriever construction)",
                    extra={"error": str(error)},
                    exc_info=True,
                )
                result.error = str(error)
                result.stage_reached = PipelineStage.FAILED
                return result

            result.stage_reached = PipelineStage.DOCUMENT_PROCESSING
            if not corpus.get_all_chunks():
                result.error = (
                    "The run corpus has no chunks "
                    f"({len(corpus.get_all_papers())} papers). Run /chunks for the "
                    "run's parsed papers before candidate gap detection. "
                    "Candidate gap detection was not run."
                )
                logger.warning(
                    "ResearchPipeline: Stage 3 — run corpus has no chunks; stopping.",
                    extra={"query_id": enriched_query.query_id, **result.corpus_stats},
                )
                return result
            logger.info(
                "ResearchPipeline: Stage 3 complete",
                extra={"query_id": enriched_query.query_id, **result.corpus_stats},
            )
        else:
            retriever = self._retriever
            result.corpus_source = CORPUS_SOURCE_INJECTED
            logger.debug(
                "ResearchPipeline: Stage 3 skipped — no corpus; using the injected retriever."
            )

        # -------------------------------------------------------------- #
        # Stage 4: Retrieval over the original + expanded queries
        # -------------------------------------------------------------- #
        try:
            chunks = self._retrieve_chunks(enriched_query, retriever, corpus, result)
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

        # Stage 5: rerank injected retrievers (RetrievalService may already
        # have reranked its scored results, in which case avoid double scoring).
        already_reranked = bool(chunks) and all(
            result.retrieval_provenance.get(chunk.chunk_id, {}).get("reranking", {}).get("status")
            in {"applied", "failed_fallback"}
            for chunk in chunks
        )
        injected_reranker = callable(getattr(type(self._reranker), "rerank", None))
        if not already_reranked and (injected_reranker or settings.RERANK_ENABLED):
            reranker = self._reranker if injected_reranker else get_cross_encoder_reranker(
                settings.RERANK_MODEL, settings.RERANK_DEVICE, settings.RERANK_BATCH_SIZE
            )
            try:
                if isinstance(reranker, CrossEncoderReranker):
                    count = min(
                        len(chunks), settings.RERANK_MAX_CANDIDATES, settings.RERANK_TOP_N
                    )
                    scores = reranker.score_pairs(
                        enriched_query.query, [chunk.text for chunk in chunks[:count]]
                    )
                    ranked = sorted(
                        zip(chunks[:count], scores), key=lambda item: item[1], reverse=True
                    )
                    chunks = [chunk for chunk, _ in ranked] + chunks[count:]
                    for rank, (chunk, score) in enumerate(ranked, start=1):
                        result.retrieval_provenance.setdefault(chunk.chunk_id, {}).update({
                            "retrieval_rank": rank,
                            "reranker_score": score,
                            "reranking": {"status": "applied", "model": reranker.model_name},
                        })
                else:
                    chunks = reranker.rerank(
                        enriched_query.query, chunks, top_n=self._retrieval_top_k
                    )
            except Exception as exc:
                if not settings.RERANK_FAIL_OPEN:
                    result.error = f"Reranking failed: {type(exc).__name__}"
                    result.stage_reached = PipelineStage.FAILED
                    return result
                logger.warning(
                    "ResearchPipeline: Stage 5 reranking failed; preserving retrieval order",
                    extra={"error_type": type(exc).__name__},
                )
                for chunk in chunks:
                    result.retrieval_provenance.setdefault(chunk.chunk_id, {})[
                        "reranking"
                    ] = {"status": "failed_fallback"}
        result.retrieved_chunks = chunks

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

        # -------------------------------------------------------------- #
        # Stage 6b (optional): pre-validation evidence-weighted scoring
        # -------------------------------------------------------------- #
        if self._evidence_scorer is not None and candidates:
            paper_lookup = corpus.get_paper_by_id if corpus is not None else None
            try:
                candidates = self._evidence_scorer.score_all(
                    list(candidates), chunks, paper_lookup
                )
            except Exception as exc:  # noqa: BLE001 — wrapped like other stage failures
                error = (
                    exc
                    if isinstance(exc, EWRGVError)
                    else GapDetectionError(
                        f"Candidate evidence scoring failed: {type(exc).__name__}: {exc}"
                    )
                )
                logger.error(
                    "ResearchPipeline: Stage 6b failed (candidate evidence scoring)",
                    extra={"error": str(error)},
                    exc_info=True,
                )
                result.candidate_gaps = list(candidates)
                result.error = str(error)
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
        result: Optional[PipelineResult] = None,
    ) -> list[PaperChunk]:
        """
        Retrieve for the original query and each expanded query, then merge.

        Merge order: best (lowest) rank across queries first, ties broken by
        the earlier query, then the earlier position.  Duplicate chunk IDs
        are collapsed.  The merged list is capped at ``retrieval_top_k``.

        Retrievers may return PaperChunks or RetrievalResults.  When
        ``corpus`` is given, every item is resolved to the corpus's
        PaperChunk by chunk_id (items not in the corpus are dropped), so
        detection always sees the canonical chunk.  Retrieval scores are
        recorded in ``result.retrieval_provenance``.
        """
        queries: list[str] = []
        for q in [query.query, *query.expanded_queries]:
            if q and q.strip() and q not in queries:
                queries.append(q)

        best: dict[str, tuple[int, int, PaperChunk, dict]] = {}
        for query_index, q in enumerate(queries):
            items = retriever.retrieve(q, top_k=self._retrieval_top_k) or []
            for position, item in enumerate(items):
                chunk, provenance = _to_paper_chunk(item, corpus)
                if chunk is None:
                    continue
                provenance["query"] = q
                key = (position + 1, query_index)
                current = best.get(chunk.chunk_id)
                if current is None or key < current[:2]:
                    best[chunk.chunk_id] = (*key, chunk, provenance)

        merged = sorted(best.values(), key=lambda entry: (entry[0], entry[1]))[
            : self._retrieval_top_k
        ]
        if result is not None:
            result.retrieval_provenance = {
                chunk.chunk_id: provenance for _, _, chunk, provenance in merged if provenance
            }
        return [chunk for _, _, chunk, _ in merged]


def _to_paper_chunk(
    item: PaperChunk | RetrievalResult, corpus: Optional[CorpusStore]
) -> tuple[Optional[PaperChunk], dict]:
    """
    Normalise a retriever output item to a PaperChunk plus score provenance.

    With a corpus, the corpus's chunk is returned (``None`` if the ID is not
    in the corpus).  Without one, a RetrievalResult is converted field by
    field (section_id is unknown and left empty).
    """
    provenance: dict = {}
    if isinstance(item, RetrievalResult):
        provenance = {
            "retrieval_score": item.score,
            "retrieval_rank": item.rank,
            "retrieval_method": item.retrieval_method,
        }
        if item.metadata:
            for key in ("reranker_score", "reranking"):
                if key in item.metadata:
                    provenance[key] = item.metadata[key]
    elif not isinstance(item, PaperChunk):
        raise TypeError(f"Unsupported retrieval result type: {type(item).__name__}")

    if corpus is not None:
        return corpus.get_chunk_by_id(item.chunk_id), provenance
    if isinstance(item, PaperChunk):
        return item, provenance

    try:
        section_type = SectionType(item.section_type)
    except ValueError:
        section_type = SectionType.UNKNOWN
    return (
        PaperChunk(
            chunk_id=item.chunk_id,
            paper_id=item.paper_id,
            section_id="",
            section_title=item.section,
            section_type=section_type,
            text=item.text,
            page_start=item.page_start or 1,
            page_end=item.page_end or item.page_start or 1,
            character_count=len(item.text),
        ),
        provenance,
    )
