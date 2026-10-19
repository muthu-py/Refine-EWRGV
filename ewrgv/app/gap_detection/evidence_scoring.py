"""
app/gap_detection/evidence_scoring.py
---------------------------------------
Evidence-weighted scoring of CANDIDATE research gaps (pre-validation).

For each CandidateGap the scorer gathers the retrieved passages the detector
cited as supporting (``relevant_passages``) and contradicting
(``contradicting_passages``), computes interpretable features from data that
is actually available, and combines them with configurable weights into a
CandidateEvidenceAssessment (score, status, features, evidence references,
explanation).

This is a prioritisation signal, NOT validation.  It only looks at evidence
the pipeline already retrieved; it does not search for counter-evidence and
it cannot establish novelty.  EWRGV validation (app/validation/) remains
responsible for deciding whether a gap is genuine.

Scoring method
--------------
Per evidence item (passage) i:

    relevance_i = |gap_terms ∩ passage_terms| / |gap_terms|
                  gap_terms = Unicode word tokens (English stop words removed)
                  of the gap statement + its key concepts; lexical, not semantic.
    quality_i   = explicit_section_quality (1.0) if the passage comes from a
                  Limitations / Future Work / Discussion / Conclusion-type
                  section, else default_section_quality (0.5).
                  Supporting evidence only (it measures how explicitly a
                  passage states the gap).
    recency_i   = max(0, 1 − max(0, reference_year − year_i) / recency_horizon_years)
                  1.0 for the reference year, 0 at horizon age or older; a
                  paper newer than the reference year (only possible with an
                  explicit reference_year) is capped at 1.0.
                  None when the passage has no publication year.
                  reference_year: the configured value if set, otherwise the
                  latest valid year across ALL chunks supplied to the scoring
                  call — one year shared by every candidate in a score_all()
                  call, so scores are comparable (no wall-clock dependence).
    strength_i  = weighted mean of the item's available features
                  (supporting: relevance, quality, recency;
                   counter: relevance, recency).

Per candidate (from SUPPORTING evidence):

    relevance    = mean relevance_i
    quality      = mean quality_i
    recency      = mean recency_i over items with a year (None if none)
    independence = min(distinct supporting papers / independence_saturation, 1)

    support_score = Σ w_f · feature_f / Σ w_f     over AVAILABLE features
    contradiction_ratio = Σ strength_counter / (Σ strength_support + Σ strength_counter)
                          (0 when both sums are 0)
    score = support_score × (1 − contradiction_penalty × contradiction_ratio)

Missing evidence: with no supporting passage the score is 0.0 and the status
INSUFFICIENT_EVIDENCE — absence of evidence is never treated as evidence that
a gap exists.  Absence of counter evidence does not raise the score either.

Status: SUPPORTED (ratio = 0, incl. zero-strength counter passages),
CONTESTED (0 < ratio < 0.5), CONTRADICTED (ratio ≥ 0.5),
INSUFFICIENT_EVIDENCE (no support).

The ``relevance`` weight must be positive: it is the only passage feature
available for every supporting and counter passage.  Counter passages get no
quality feature, so a passage weighs somewhat more as counter evidence than
the same passage would as support (a deliberately conservative bias).

Not used: citation counts (a popularity proxy confounded with paper age and
often missing) — reported per item for information only when available.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Optional

from app.core.logging import get_logger
from app.domain.enums import CandidateEvidenceStatus, EvidenceType
from app.domain.models.candidate_evidence import (
    CandidateEvidenceAssessment,
    CandidateEvidenceItem,
)
from app.domain.models.candidate_gap import CandidateGap
from app.domain.models.paper import DocumentChunk, Paper
from app.gap_detection.ranking import _normalise_statement, _statement_tokens

logger = get_logger(__name__)

FEATURES = ("relevance", "quality", "recency", "independence")

DEFAULT_WEIGHTS: dict[str, float] = {
    "relevance": 0.35,
    "quality": 0.20,
    "recency": 0.15,
    "independence": 0.30,
}

# Section headings (casefolded, matched as substrings) whose passages state
# limitations / open problems explicitly.
DEFAULT_EXPLICIT_SECTIONS: tuple[str, ...] = (
    "limitation",
    "future work",
    "future direction",
    "discussion",
    "conclusion",
    "open problem",
    "open challenge",
    "threats to validity",
)

_CONTESTED_THRESHOLD = 0.5


def _terms(text: str) -> frozenset[str]:
    return _statement_tokens(_normalise_statement(text))


def _weighted_mean(values: dict[str, Optional[float]], weights: dict[str, float]) -> float:
    used = {f: w for f, w in weights.items() if values.get(f) is not None and w > 0}
    total = sum(used.values())
    if total == 0:
        return 0.0
    return sum(w * values[f] for f, w in used.items()) / total  # type: ignore[operator]


def _mean(values: Iterable[float]) -> Optional[float]:
    items = list(values)
    return sum(items) / len(items) if items else None


class CandidateEvidenceScorer:
    """
    Deterministic, configurable evidence-weighted scorer for CandidateGaps.

    Parameters
    ----------
    weights:
        Non-negative feature weights (any subset of ``FEATURES``; missing
        features get weight 0; ``relevance`` must be positive).  Renormalised
        over the features available for each candidate.  Defaults to
        ``DEFAULT_WEIGHTS``.
    independence_saturation:
        Number of distinct supporting papers at which independence = 1.
    recency_horizon_years:
        Age (relative to the reference year) at which recency reaches 0.
    reference_year:
        Positive year against which recency is measured.  ``None`` (default)
        uses the latest valid publication year across all chunks supplied to
        the scoring call, shared by every candidate in that call.
    contradiction_penalty:
        In [0, 1]; how strongly the contradiction ratio reduces the score.
        0 disables the penalty (counter evidence is still reported).
    explicit_section_quality, default_section_quality:
        Quality values for explicit-limitation sections vs. any other section.
    explicit_sections:
        Casefolded substrings identifying explicit-limitation sections.
    """

    def __init__(
        self,
        weights: Optional[dict[str, float]] = None,
        independence_saturation: int = 3,
        recency_horizon_years: int = 10,
        reference_year: Optional[int] = None,
        contradiction_penalty: float = 1.0,
        explicit_section_quality: float = 1.0,
        default_section_quality: float = 0.5,
        explicit_sections: tuple[str, ...] = DEFAULT_EXPLICIT_SECTIONS,
    ) -> None:
        raw = dict(DEFAULT_WEIGHTS if weights is None else weights)
        unknown = set(raw) - set(FEATURES)
        if unknown:
            raise ValueError(f"Unknown feature weights: {sorted(unknown)}; valid: {FEATURES}")
        if any(w < 0 or math.isnan(w) for w in raw.values()) or sum(raw.values()) <= 0:
            raise ValueError("Feature weights must be non-negative and not all zero.")
        if raw.get("relevance", 0.0) <= 0:
            raise ValueError(
                "The 'relevance' weight must be positive: relevance is the only passage-level "
                "feature available for every supporting and counter-evidence passage, so "
                "without it passage strengths (and the contradiction ratio) can be 0 or "
                "undefined (e.g. weights={'independence': 1.0} or {'quality': 1.0})."
            )
        if reference_year is not None and (
            isinstance(reference_year, bool)
            or not isinstance(reference_year, int)
            or reference_year <= 0
        ):
            raise ValueError("reference_year must be a positive integer year or None.")
        if independence_saturation < 1:
            raise ValueError("independence_saturation must be >= 1.")
        if recency_horizon_years < 1:
            raise ValueError("recency_horizon_years must be >= 1.")
        if not 0.0 <= contradiction_penalty <= 1.0:
            raise ValueError("contradiction_penalty must be in [0, 1].")
        for value in (explicit_section_quality, default_section_quality):
            if not 0.0 <= value <= 1.0:
                raise ValueError("Section quality values must be in [0, 1].")

        self._weights = {f: float(raw.get(f, 0.0)) for f in FEATURES}
        self._independence_saturation = independence_saturation
        self._recency_horizon = recency_horizon_years
        self._reference_year = reference_year
        self._contradiction_penalty = contradiction_penalty
        self._explicit_quality = explicit_section_quality
        self._default_quality = default_section_quality
        self._explicit_sections = tuple(s.casefold() for s in explicit_sections)

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def score_all(
        self,
        gaps: list[CandidateGap],
        chunks: list[DocumentChunk],
        paper_lookup: Optional[Callable[[str], Optional[Paper]]] = None,
    ) -> list[CandidateGap]:
        """
        Return copies of ``gaps`` with ``evidence_assessment`` set (inputs unchanged).

        One reference year is resolved for the whole call (see
        ``resolve_reference_year``) and shared by every candidate, so recency —
        and therefore scores — are comparable across the candidates.
        """
        chunks_by_id = {c.chunk_id: c for c in chunks}
        reference_year = self.resolve_reference_year(chunks, paper_lookup)
        return [
            gap.model_copy(
                update={
                    "evidence_assessment": self.score(
                        gap, chunks_by_id, paper_lookup, reference_year=reference_year
                    )
                }
            )
            for gap in gaps
        ]

    def resolve_reference_year(
        self,
        chunks: Iterable[DocumentChunk],
        paper_lookup: Optional[Callable[[str], Optional[Paper]]] = None,
    ) -> Optional[int]:
        """
        The configured ``reference_year`` if set; otherwise the latest valid
        publication year across ALL supplied chunks (chunk metadata, else the
        paper lookup).  ``None`` when no chunk has a year.
        """
        if self._reference_year is not None:
            return self._reference_year
        years = (self._year(c, paper_lookup) for c in chunks)
        return max((y for y in years if y is not None), default=None)

    def score(
        self,
        gap: CandidateGap,
        chunks_by_id: dict[str, DocumentChunk],
        paper_lookup: Optional[Callable[[str], Optional[Paper]]] = None,
        reference_year: Optional[int] = None,
    ) -> CandidateEvidenceAssessment:
        """
        Score one candidate against the available (retrieved) chunks.

        ``reference_year`` is normally supplied by ``score_all``; when omitted
        it is resolved from all of ``chunks_by_id`` (same rule), never from
        this candidate's evidence alone.
        """
        if reference_year is None:
            reference_year = self.resolve_reference_year(chunks_by_id.values(), paper_lookup)
        gap_terms = self._gap_terms(gap)
        missing: list[str] = []

        def collect(chunk_ids: list[str], exclude: set[str]) -> list[DocumentChunk]:
            found: list[DocumentChunk] = []
            for cid in dict.fromkeys(chunk_ids):
                if cid in exclude:
                    continue
                chunk = chunks_by_id.get(cid)
                if chunk is None:
                    missing.append(cid)
                else:
                    found.append(chunk)
            return found

        counter_ids = set(gap.contradicting_passages)
        supporting_chunks = collect(gap.relevant_passages, exclude=counter_ids)
        counter_chunks = collect(gap.contradicting_passages, exclude=set())

        years = {
            c.chunk_id: self._year(c, paper_lookup) for c in supporting_chunks + counter_chunks
        }

        supporting = [
            self._item(c, EvidenceType.SUPPORTING, gap_terms, years[c.chunk_id],
                       reference_year, paper_lookup)
            for c in supporting_chunks
        ]
        counter = [
            self._item(c, EvidenceType.COUNTER, gap_terms, years[c.chunk_id],
                       reference_year, paper_lookup)
            for c in counter_chunks
        ]

        if not supporting:
            features: dict[str, Optional[float]] = dict.fromkeys(FEATURES)
            assessment = CandidateEvidenceAssessment(
                score=0.0,
                status=CandidateEvidenceStatus.INSUFFICIENT_EVIDENCE,
                support_score=0.0,
                contradiction_ratio=1.0 if any(i.strength > 0 for i in counter) else 0.0,
                features=features,
                weights={},
                supporting=[],
                contradicting=counter,
                missing_chunk_ids=missing,
            )
            return assessment.model_copy(update={"explanation": self._explain(assessment)})

        features = {
            "relevance": _mean(i.relevance for i in supporting),
            "quality": _mean(i.quality for i in supporting if i.quality is not None),
            "recency": _mean(i.recency for i in supporting if i.recency is not None),
            "independence": min(
                len({i.paper_id for i in supporting}) / self._independence_saturation, 1.0
            ),
        }
        used = {f: w for f, w in self._weights.items() if features[f] is not None and w > 0}
        total = sum(used.values())
        effective_weights = {f: w / total for f, w in used.items()} if total else {}
        support_score = _weighted_mean(features, self._weights)

        support_mass = sum(i.strength for i in supporting)
        counter_mass = sum(i.strength for i in counter)
        mass = support_mass + counter_mass
        # No measurable evidence strength on either side → no contradiction.
        ratio = counter_mass / mass if mass > 0 else 0.0
        score = support_score * (1.0 - self._contradiction_penalty * ratio)

        # Status follows the contradiction ratio, not the mere presence of
        # counter passages: zero-strength counter evidence has no effect.
        if ratio == 0.0:
            status = CandidateEvidenceStatus.SUPPORTED
        elif ratio < _CONTESTED_THRESHOLD:
            status = CandidateEvidenceStatus.CONTESTED
        else:
            status = CandidateEvidenceStatus.CONTRADICTED

        assessment = CandidateEvidenceAssessment(
            score=round(min(max(score, 0.0), 1.0), 6),
            status=status,
            support_score=round(support_score, 6),
            contradiction_ratio=round(ratio, 6),
            features={f: (None if v is None else round(v, 6)) for f, v in features.items()},
            weights={f: round(w, 6) for f, w in effective_weights.items()},
            supporting=supporting,
            contradicting=counter,
            missing_chunk_ids=missing,
        )
        return assessment.model_copy(update={"explanation": self._explain(assessment)})

    # ------------------------------------------------------------------ #
    # Features
    # ------------------------------------------------------------------ #

    @staticmethod
    def _gap_terms(gap: CandidateGap) -> frozenset[str]:
        parts = [gap.gap_statement]
        if gap.structured_representation is not None:
            parts.extend(gap.structured_representation.key_concepts)
        return _terms(" ".join(parts))

    @staticmethod
    def _year(
        chunk: DocumentChunk, paper_lookup: Optional[Callable[[str], Optional[Paper]]]
    ) -> Optional[int]:
        year = chunk.metadata.get("publication_year")
        if isinstance(year, int) and not isinstance(year, bool):
            return year
        if paper_lookup is not None:
            paper = paper_lookup(chunk.paper_id)
            if paper is not None:
                if paper.publication_year:
                    return paper.publication_year
                if paper.publication_date:
                    return paper.publication_date.year
        return None

    def _quality(self, section: str) -> float:
        heading = (section or "").casefold()
        if any(marker in heading for marker in self._explicit_sections):
            return self._explicit_quality
        return self._default_quality

    def _recency(self, year: Optional[int], reference_year: Optional[int]) -> Optional[float]:
        if year is None or reference_year is None:
            return None
        age = max(reference_year - year, 0)
        return max(0.0, 1.0 - age / self._recency_horizon)

    def _item(
        self,
        chunk: DocumentChunk,
        stance: EvidenceType,
        gap_terms: frozenset[str],
        year: Optional[int],
        reference_year: Optional[int],
        paper_lookup: Optional[Callable[[str], Optional[Paper]]],
    ) -> CandidateEvidenceItem:
        passage_terms = _terms(chunk.text)
        relevance = len(gap_terms & passage_terms) / len(gap_terms) if gap_terms else 0.0
        quality = self._quality(chunk.section) if stance == EvidenceType.SUPPORTING else None
        recency = self._recency(year, reference_year)
        item_weights = {f: self._weights[f] for f in ("relevance", "quality", "recency")}
        strength = _weighted_mean(
            {"relevance": relevance, "quality": quality, "recency": recency}, item_weights
        )
        paper = paper_lookup(chunk.paper_id) if paper_lookup is not None else None
        return CandidateEvidenceItem(
            chunk_id=chunk.chunk_id,
            paper_id=chunk.paper_id,
            section=chunk.section,
            stance=stance,
            relevance=round(relevance, 6),
            quality=quality,
            recency=None if recency is None else round(recency, 6),
            strength=round(strength, 6),
            publication_year=year,
            title=chunk.metadata.get("title") or (paper.title if paper else None),
            doi=chunk.metadata.get("doi") or (paper.doi if paper else None),
            citation_count=paper.citation_count if paper else None,
        )

    # ------------------------------------------------------------------ #
    # Explanation
    # ------------------------------------------------------------------ #

    def _explain(self, a: CandidateEvidenceAssessment) -> str:
        caveat = (
            "Candidate-level evidence score from retrieved passages only; "
            "not a validation of the gap."
        )
        missing = (
            f" {len(a.missing_chunk_ids)} cited passage(s) were not available and were ignored."
            if a.missing_chunk_ids else ""
        )
        effective = [i for i in a.contradicting if i.strength > 0]
        ineffective = len(a.contradicting) - len(effective)
        no_effect = (
            f" {ineffective} contradicting passage(s) had zero strength (no gap terms in "
            "common and no positive recency) and did not affect the assessment."
            if ineffective else ""
        )
        if a.status == CandidateEvidenceStatus.INSUFFICIENT_EVIDENCE:
            counter = (
                f" {len(effective)} passage(s) suggest the gap may already be addressed."
                if effective else ""
            )
            return (
                "Evidence score 0.00 (insufficient_evidence): no retrieved passage supports "
                "this gap. Missing evidence is not evidence that the gap exists."
                f"{counter}{no_effect}{missing} {caveat}"
            )

        papers = len({i.paper_id for i in a.supporting})
        parts = []
        for name in FEATURES:
            value = a.features.get(name)
            if value is None:
                parts.append(f"{name} unavailable")
            else:
                parts.append(f"{name} {value:.2f} (weight {a.weights.get(name, 0.0):.2f})")
        text = (
            f"Evidence score {a.score:.2f} ({a.status.value}): "
            f"{len(a.supporting)} supporting passage(s) from {papers} independent paper(s); "
            f"support score {a.support_score:.2f} from " + ", ".join(parts) + "."
        )
        if effective:
            reduction = a.support_score - a.score
            text += (
                f" {len(effective)} contradicting passage(s) carry "
                f"{a.contradiction_ratio:.0%} of the evidence strength and reduce the score "
                f"by {reduction:.2f}."
            )
        elif not a.contradicting:
            text += " No contradicting passage was retrieved (this does not confirm the gap)."
        return text + no_effect + missing + " " + caveat
