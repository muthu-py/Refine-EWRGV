"""
app/gap_detection/detector.py
-------------------------------
Gap detector: identifies candidate research gaps from retrieved evidence.

The detector satisfies the GapDetector interface (app.domain.interfaces).
It receives retrieved DocumentChunks and a ResearchQuery, and produces
a list of CandidateGap domain objects.

The detected gaps are *unvalidated*.  All downstream confidence and
classification is the responsibility of the EWRGV validation module.

Flow
----
    ResearchQuery + DocumentChunk[]
              ↓
    build_gap_detection_prompt()          ← prompts.py
              ↓
    LLMProvider.complete_structured()     (one call for all candidates)
              ↓
    GapDetectionLLMOutput (validation)    ← schemas.py
              ↓
    grounding check: cited chunk IDs must be among the input chunks
              ↓
    CandidateGap[] (+ StructuredGapRepresentation via representation.py)

Error handling
--------------
Provider failures, non-dict responses, and an invalid top-level response
are raised as GapDetectionError (via llm_call.call_structured_llm, which
keeps the redacted provider cause in the message).  Individually malformed
candidates are dropped and logged; if every candidate is malformed the
response is treated as invalid and GapDetectionError is raised.
"""

from __future__ import annotations

import re

import pydantic

from app.core.exceptions import GapDetectionError
from app.core.logging import get_logger
from app.domain.enums import GapType
from app.domain.interfaces import LLMProvider
from app.domain.models.candidate_gap import CandidateGap
from app.domain.models.paper import DocumentChunk
from app.domain.models.query import ResearchQuery
from app.gap_detection.llm_call import call_structured_llm
from app.gap_detection.prompts import build_gap_detection_prompt
from app.gap_detection.representation import GapRepresentationBuilder
from app.gap_detection.schemas import CandidateGapLLMOutput, GapDetectionLLMOutput

logger = get_logger(__name__)

# Defaults; MAX_CANDIDATE_GAPS in settings is the usual source for max_candidates.
_DEFAULT_MAX_CANDIDATES = 20
_DEFAULT_MAX_CHUNKS = 30
_DEFAULT_MAX_CHUNK_CHARS = 1500

# Lenient aliases for gap-type labels the LLM may return instead of the
# exact enum values.  Keys are casefolded.
_GAP_TYPE_ALIASES: dict[str, GapType] = {
    **{t.value.casefold(): t for t in GapType},
    **{t.name.casefold(): t for t in GapType},
    "contradictory": GapType.CONTRADICTORY_EVIDENCE,
    "contradiction": GapType.CONTRADICTORY_EVIDENCE,
    "methodology": GapType.METHODOLOGICAL,
    "method": GapType.METHODOLOGICAL,
    "data": GapType.DATASET,
    "other": GapType.OTHER,
    "unknown": GapType.OTHER,
}


def map_gap_type(label: str | None) -> GapType | None:
    """Map a raw LLM label to a GapType, or None if unrecognised."""
    if not label:
        return None
    return _GAP_TYPE_ALIASES.get(label.strip().casefold())


# Harmless decoration the LLM may copy from the prompt's "[chunk_id: X]"
# passage headers: surrounding brackets/quotes/backticks and a "chunk_id:" label.
_CHUNK_ID_WRAPPERS = "[](){}<>\"'`"
_CHUNK_ID_LABEL_RE = re.compile(r"^(?:chunk[\s_-]?id|id)\s*[:=]\s*", re.IGNORECASE)


def resolve_chunk_id(cited: str, supplied: dict[str, DocumentChunk] | set[str]) -> str | None:
    """
    Resolve a cited chunk ID against the IDs supplied to the LLM.

    An exact match always wins.  Otherwise harmless decoration is stripped
    (brackets, quotes, backticks, a "chunk_id:" label) and the result must
    match a supplied ID *exactly* (case-sensitive).  IDs that were not
    supplied are never accepted.
    """
    if cited in supplied:
        return cited
    candidate = cited.strip()
    for _ in range(3):  # peel up to a few nested layers, e.g. "[`chunk_id: c1`]"
        previous = candidate
        candidate = candidate.strip(_CHUNK_ID_WRAPPERS).strip()
        candidate = _CHUNK_ID_LABEL_RE.sub("", candidate).strip()
        if candidate == previous:
            break
    return candidate if candidate and candidate in supplied else None


class GapDetector:
    """
    Identifies candidate research gaps from retrieved evidence chunks.

    Parameters
    ----------
    llm_provider:
        Any object satisfying the LLMProvider protocol.
    max_candidates:
        Upper bound on returned candidates (typically settings.MAX_CANDIDATE_GAPS).
    max_chunks:
        Only the first ``max_chunks`` unique input chunks (in retrieval
        order) are sent to the LLM, to bound prompt size.
    max_chunk_chars:
        Per-passage character cap in the prompt.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        max_candidates: int = _DEFAULT_MAX_CANDIDATES,
        max_chunks: int = _DEFAULT_MAX_CHUNKS,
        max_chunk_chars: int = _DEFAULT_MAX_CHUNK_CHARS,
    ) -> None:
        if max_candidates < 1 or max_chunks < 1 or max_chunk_chars < 1:
            raise ValueError("max_candidates, max_chunks and max_chunk_chars must be >= 1.")
        self._llm = llm_provider
        self._max_candidates = max_candidates
        self._max_chunks = max_chunks
        self._max_chunk_chars = max_chunk_chars

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def detect(
        self,
        query: ResearchQuery,
        chunks: list[DocumentChunk],
    ) -> list[CandidateGap]:
        """
        Detect candidate gaps from retrieved chunks.

        Returns an unranked list of CandidateGap objects.  Returns an empty
        list without calling the LLM when ``chunks`` is empty.

        Candidates whose cited chunk IDs are all absent from the input are
        discarded as ungrounded; invalid IDs on otherwise-grounded
        candidates are removed and recorded in ``metadata["invalid_chunk_ids"]``.

        Raises
        ------
        GapDetectionError
            On provider failure or invalid LLM output.
        """
        selected = self._select_chunks(chunks)
        if not selected:
            logger.info(
                "GapDetector: no chunks provided; skipping detection",
                extra={"query_id": query.query_id},
            )
            return []

        prompt = build_gap_detection_prompt(
            query,
            selected,
            max_gaps=self._max_candidates,
            max_chunk_chars=self._max_chunk_chars,
        )
        raw = call_structured_llm(
            self._llm,
            prompt,
            GapDetectionLLMOutput.json_schema_for_llm(),
            operation="gap detection",
            log_extra={"query_id": query.query_id},
        )
        output = self._validate_llm_output(raw, query.query_id)

        chunks_by_id = {c.chunk_id: c for c in selected}
        gaps: list[CandidateGap] = []
        dropped = 0
        for item in output.candidate_gaps:
            gap = self._map_to_domain(item, chunks_by_id, query.query_id)
            if gap is None:
                dropped += 1
                continue
            gaps.append(gap)
            if len(gaps) >= self._max_candidates:
                break

        logger.info(
            "GapDetector: detection complete",
            extra={
                "query_id": query.query_id,
                "chunks_analysed": len(selected),
                "candidates_returned": len(gaps),
                "ungrounded_dropped": dropped,
            },
        )
        return gaps

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _select_chunks(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """De-duplicate by chunk_id (first occurrence wins) and cap the count."""
        seen: set[str] = set()
        selected: list[DocumentChunk] = []
        for chunk in chunks or []:
            if chunk.chunk_id in seen or not chunk.text.strip():
                continue
            seen.add(chunk.chunk_id)
            selected.append(chunk)
            if len(selected) >= self._max_chunks:
                break
        return selected

    def _validate_llm_output(self, raw: dict, query_id: str) -> GapDetectionLLMOutput:
        """
        Validate the response.  Top-level problems raise; individually
        malformed candidates are dropped (and logged) unless every candidate
        is malformed, which indicates a broken response and raises.
        """
        try:
            output = GapDetectionLLMOutput.model_validate(raw)
        except pydantic.ValidationError as exc:
            logger.warning(
                "GapDetector: LLM output failed validation",
                extra={"query_id": query_id, "errors": exc.error_count()},
            )
            raise GapDetectionError(
                "LLM returned invalid structured output for gap detection.",
                detail=str(exc),
            ) from exc

        if output.dropped_candidates:
            if not output.candidate_gaps:
                raise GapDetectionError(
                    "LLM returned no valid candidate gaps; every candidate was malformed.",
                    detail="; ".join(output.dropped_candidates),
                )
            logger.warning(
                "GapDetector: dropped %d malformed candidate(s)",
                len(output.dropped_candidates),
                extra={"query_id": query_id, "dropped": output.dropped_candidates},
            )
        return output

    def _map_to_domain(
        self,
        item: CandidateGapLLMOutput,
        chunks_by_id: dict[str, DocumentChunk],
        query_id: str,
    ) -> CandidateGap | None:
        """Map one validated LLM candidate to a CandidateGap, or None if ungrounded."""
        valid_ids: list[str] = []
        invalid_ids: list[str] = []
        for cited in item.supporting_chunk_ids:
            cid = resolve_chunk_id(cited, chunks_by_id)
            if cid is not None:
                if cid not in valid_ids:
                    valid_ids.append(cid)
            elif cited not in invalid_ids:
                invalid_ids.append(cited)

        if not valid_ids:
            logger.warning(
                "GapDetector: dropping candidate with no valid supporting chunk IDs",
                extra={"query_id": query_id, "invalid_chunk_ids": invalid_ids},
            )
            return None

        source_papers: list[str] = []
        for cid in valid_ids:
            pid = chunks_by_id[cid].paper_id
            if pid not in source_papers:
                source_papers.append(pid)

        metadata: dict = {"detector": "llm", "query_id": query_id}
        primary = map_gap_type(item.gap_type)
        if primary is None:
            primary = GapType.OTHER
            if item.gap_type:
                metadata["raw_gap_type"] = item.gap_type

        secondary: list[GapType] = []
        for label in item.secondary_gap_types:
            mapped = map_gap_type(label)
            if mapped is not None and mapped != primary and mapped not in secondary:
                secondary.append(mapped)

        if invalid_ids:
            metadata["invalid_chunk_ids"] = invalid_ids
        if item.rationale:
            metadata["rationale"] = item.rationale

        return CandidateGap(
            gap_statement=item.gap_statement,
            primary_gap_type=primary,
            secondary_gap_types=secondary,
            structured_representation=GapRepresentationBuilder.from_llm_output(
                item.gap_statement, item
            ),
            source_papers=source_papers,
            relevant_passages=valid_ids,
            initial_confidence=item.initial_confidence,
            metadata=metadata,
        )
