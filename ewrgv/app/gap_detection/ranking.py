"""
app/gap_detection/ranking.py
-----------------------------
Candidate gap ranking.

Takes a list of unranked CandidateGap objects and produces the Top-K list
passed to EWRGV validation.

Heuristic (deterministic, no LLM)
---------------------------------
1. Base score per candidate:

       base = w_confidence * initial_confidence
            + w_papers     * min(distinct_source_papers / paper_saturation, 1)
            + w_passages   * min(distinct_passages      / passage_saturation, 1)

   Weights are normalised to sum to 1, so base ∈ [0, 1].

2. Near-duplicate removal: candidates are visited in base-score order; a
   candidate whose statement is identical (ignoring case, punctuation and
   Unicode normalisation form) to, or has Unicode-aware token Jaccard
   similarity ≥ ``duplicate_threshold`` with, an already-kept candidate is
   dropped and its gap_id recorded on the kept one
   (``metadata["ranking"]["merged_duplicates"]``).  NaN confidence is
   scored as 0.

3. Gap-type diversity (greedy selection): at each step the candidate with
   the highest

       adjusted = base - diversity_penalty * (already-selected with same primary type)

   is chosen.  Ties break on base score, then statement text, so the
   output is fully deterministic.

What this heuristic does NOT measure
-------------------------------------
It does not measure novelty or whether a gap exists in the wider
literature.  ``initial_confidence`` is the detector LLM's self-reported
grounding strength, and evidence breadth only reflects the passages the
detector cited.  Establishing whether a gap is genuine is the job of EWRGV
validation.
"""

from __future__ import annotations

import math
import unicodedata

from app.core.logging import get_logger
from app.domain.models.candidate_gap import CandidateGap

logger = get_logger(__name__)

_STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
        "have", "in", "is", "it", "its", "no", "not", "of", "on", "or", "that",
        "the", "their", "there", "this", "to", "was", "were", "which", "with",
    }
)


def _normalise_statement(statement: str) -> str:
    """NFKC-normalise and casefold, so composed/decomposed accents compare equal."""
    return unicodedata.normalize("NFKC", statement).casefold()


def _statement_tokens(normalised: str) -> frozenset[str]:
    """
    Split a normalised statement into word tokens, Unicode-aware.

    A token is a run of letters (L*), combining marks (M*, needed for
    scripts such as Tamil and Devanagari), and digits (N*).  Accented Latin
    words stay whole ("modèles").  Scripts written without spaces (e.g.
    Chinese) yield one token per run, so only identical runs match.
    """
    tokens: list[str] = []
    current: list[str] = []
    for ch in normalised:
        if unicodedata.category(ch)[0] in "LMN":
            current.append(ch)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return frozenset(t for t in tokens if t not in _STOP_WORDS)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _alphanumeric_key(normalised: str) -> str:
    """Statement with punctuation, whitespace and symbols removed."""
    return "".join(ch for ch in normalised if unicodedata.category(ch)[0] in "LMN")


class CandidateRanker:
    """
    Ranks CandidateGap objects to select the Top-K for EWRGV validation.

    Parameters
    ----------
    confidence_weight, paper_weight, passage_weight:
        Non-negative weights for the base score components (normalised).
    paper_saturation, passage_saturation:
        Counts at which the evidence-breadth components reach their maximum.
    duplicate_threshold:
        Token Jaccard similarity in (0, 1] at or above which two statements
        are treated as near-duplicates.
    diversity_penalty:
        Score penalty per already-selected candidate of the same primary
        gap type.  0 disables diversity re-ranking.
    """

    def __init__(
        self,
        confidence_weight: float = 0.6,
        paper_weight: float = 0.25,
        passage_weight: float = 0.15,
        paper_saturation: int = 3,
        passage_saturation: int = 5,
        duplicate_threshold: float = 0.8,
        diversity_penalty: float = 0.1,
    ) -> None:
        weights = (confidence_weight, paper_weight, passage_weight)
        if any(w < 0 for w in weights) or sum(weights) <= 0:
            raise ValueError("Ranking weights must be non-negative and not all zero.")
        if paper_saturation < 1 or passage_saturation < 1:
            raise ValueError("Saturation values must be >= 1.")
        if not 0.0 < duplicate_threshold <= 1.0:
            raise ValueError("duplicate_threshold must be in (0, 1].")
        if diversity_penalty < 0:
            raise ValueError("diversity_penalty must be non-negative.")

        total = sum(weights)
        self._w_conf = confidence_weight / total
        self._w_papers = paper_weight / total
        self._w_passages = passage_weight / total
        self._paper_saturation = paper_saturation
        self._passage_saturation = passage_saturation
        self._duplicate_threshold = duplicate_threshold
        self._diversity_penalty = diversity_penalty

    def _is_near_duplicate(
        self,
        a: tuple[str, frozenset[str]],
        b: tuple[str, frozenset[str]],
    ) -> bool:
        """
        Two statements are duplicates when their alphanumeric content is
        identical, or when both have tokens and their Jaccard similarity
        reaches the threshold.  Token-less statements are never merged
        unless identical, so unusual text cannot collapse distinct gaps.
        """
        key_a, tokens_a = a
        key_b, tokens_b = b
        if key_a and key_a == key_b:
            return True
        if not tokens_a or not tokens_b:
            return False
        return _jaccard(tokens_a, tokens_b) >= self._duplicate_threshold

    def base_score(self, gap: CandidateGap) -> float:
        """Return the base score in [0, 1] before duplicate and diversity handling."""
        # NaN would make every comparison False and corrupt the sort; treat
        # it as no confidence.  ±inf is clamped like any out-of-range value.
        confidence = 0.0 if math.isnan(gap.initial_confidence) else gap.initial_confidence
        confidence = min(max(confidence, 0.0), 1.0)
        papers = len(set(gap.source_papers))
        passages = len(set(gap.relevant_passages))
        return (
            self._w_conf * confidence
            + self._w_papers * min(papers / self._paper_saturation, 1.0)
            + self._w_passages * min(passages / self._passage_saturation, 1.0)
        )

    def rank(
        self,
        candidates: list[CandidateGap],
        top_k: int = 5,
    ) -> list[CandidateGap]:
        """
        Rank candidates and return the top-k as new CandidateGap copies.

        Returned copies have ``rank`` set (1 = highest priority) and ranking
        details under ``metadata["ranking"]``.  Input objects are not mutated.
        """
        if top_k < 1 or not candidates:
            return []

        scored = sorted(
            ((self.base_score(g), g) for g in candidates),
            key=lambda pair: (-pair[0], pair[1].gap_statement.casefold(), pair[1].gap_id),
        )

        # --- Near-duplicate removal (keep the higher-scored candidate) ---
        kept: list[tuple[float, CandidateGap, tuple[str, frozenset[str]], list[str]]] = []
        for score, gap in scored:
            normalised = _normalise_statement(gap.gap_statement)
            signature = (_alphanumeric_key(normalised), _statement_tokens(normalised))
            duplicate_of = next(
                (entry for entry in kept if self._is_near_duplicate(signature, entry[2])),
                None,
            )
            if duplicate_of is not None:
                duplicate_of[3].append(gap.gap_id)
                continue
            kept.append((score, gap, signature, []))

        # --- Greedy selection with gap-type diversity penalty ---
        remaining = list(kept)
        type_counts: dict = {}
        ranked: list[CandidateGap] = []

        def adjusted(entry: tuple) -> float:
            return entry[0] - self._diversity_penalty * type_counts.get(
                entry[1].primary_gap_type, 0
            )

        while remaining and len(ranked) < top_k:
            best = min(
                remaining,
                key=lambda e: (-adjusted(e), -e[0], e[1].gap_statement.casefold(), e[1].gap_id),
            )
            remaining.remove(best)
            score, gap, _, merged = best
            penalty = score - adjusted(best)
            type_counts[gap.primary_gap_type] = type_counts.get(gap.primary_gap_type, 0) + 1

            metadata = dict(gap.metadata)
            metadata["ranking"] = {
                "base_score": round(score, 6),
                "diversity_penalty": round(penalty, 6),
                "adjusted_score": round(score - penalty, 6),
                "distinct_papers": len(set(gap.source_papers)),
                "distinct_passages": len(set(gap.relevant_passages)),
                "merged_duplicates": list(merged),
            }
            ranked.append(
                gap.model_copy(update={"rank": len(ranked) + 1, "metadata": metadata})
            )

        logger.info(
            "CandidateRanker: ranking complete",
            extra={
                "candidates_in": len(candidates),
                "after_dedup": len(kept),
                "returned": len(ranked),
            },
        )
        return ranked
