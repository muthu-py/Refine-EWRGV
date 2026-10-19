"""
app/explanation/generator.py
------------------------------
Explanation generator: a natural-language narrative of the EWRGV result.

``TemplateExplanationGenerator`` (used by EWRGV) writes the explanation from
the evidence chain deterministically - every claim in it is a number or a
citation from the validation run, so it cannot hallucinate.
``LLMExplanationGenerator`` is kept as an optional, unimplemented alternative.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.models.candidate_gap import CandidateGap
from app.domain.models.validation import EvidenceChainEntry, ValidationResult

logger = get_logger(__name__)

_CLASS_TEXT = {
    "VALID_GAP": "is supported as a genuine gap",
    "UNCERTAIN_GAP": "remains uncertain",
    "UNSUPPORTED_GAP": "is not supported",
}
_REASON_TEXT = {
    "SUPPORTED": "the stating papers and supporting evidence clearly outweigh work that addresses it",
    "ADDRESSED_EXTERNALLY": "the validation searches found work that already addresses it",
    "CORPUS_ARTEFACT": "it looks sparse only in this run's corpus; the wider literature covers it",
    "CONTESTED": "counter evidence is substantial relative to the support",
    "INSUFFICIENT_EVIDENCE": "too little relevant evidence was found to judge it",
    "WEAK_SUPPORT": "its support is limited",
}


class TemplateExplanationGenerator:
    def generate(self, gap: dict, classification: str, reason: str, confidence: float,
                 chain: list[EvidenceChainEntry]) -> str:
        counts = {}
        for entry in chain:
            if entry.label.endswith("evidence"):
                counts[entry.label.split()[0].lower()] = len(entry.evidence_ids)
        parts = [f"The candidate gap \"{gap['gap_statement'][:160]}\" {_CLASS_TEXT.get(classification, classification)} "
                 f"(confidence {confidence:.2f}): {_REASON_TEXT.get(reason, reason)}."]
        if counts:
            parts.append("Evidence: " + ", ".join(f"{n} {k}" for k, n in counts.items()) + ".")
        cov = next((e.summary for e in chain if e.label == "Coverage"), None)
        if cov:
            parts.append(f"Coverage: {cov}.")
        return " ".join(parts)


class LLMExplanationGenerator:
    """Optional LLM narrative (not implemented; the template generator is used)."""

    def __init__(self, llm_provider: object) -> None:
        self._llm = llm_provider

    def generate(self, gap: CandidateGap, result: ValidationResult) -> str:
        logger.warning("LLMExplanationGenerator is not implemented; use TemplateExplanationGenerator.")
        return (
            f"[PLACEHOLDER] Gap '{gap.gap_statement[:60]}...' "
            f"classified as {result.classification} "
            f"(confidence: {result.confidence})."
        )
