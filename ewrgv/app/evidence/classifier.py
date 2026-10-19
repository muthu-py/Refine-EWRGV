"""
app/evidence/classifier.py
---------------------------
Evidence classifier: assigns an EvidenceType to a retrieved item (run-corpus
chunk or external title + abstract) relative to a candidate gap.

``RuleEvidenceClassifier`` (used by EWRGV) combines embeddings and rules,
deterministic and free:

    relevance  = cosine(gap statement, item text)               (bge)
    coverage   = share of the gap's concepts (or any alias) named in the item
    states     = the item contains a limitation / open-problem sentence
                 (app.knowledge_graph.statements) about the gap
    solves     = the item proposes / reports a solution
                 (propose, present, introduce, novel, achieve, outperform, ...)
    widely     = "has been widely / extensively studied ..."

    relevance < 0.45                                     -> INSUFFICIENT
    widely and coverage >= 0.5                           -> CONTRADICTORY  (the claim "X is missing" is false)
    solves and coverage == 1 and relevance >= 0.62       -> COUNTER        (the gap is addressed)
    solves and (coverage >= 0.5 or relevance >= 0.62)    -> PARTIAL
    states and (coverage >= 0.5 or relevance >= 0.55)    -> SUPPORTING     (another paper reports the same gap)
    otherwise                                            -> INSUFFICIENT

A paper that both states the limitation and proposes a solution to it is
counter evidence (it is the work that fills the gap), so solution rules come
first.  ``LLMEvidenceClassifier`` is kept as an optional alternative
(not implemented; not needed for EWRGV).
"""

from __future__ import annotations

import math
import re
from typing import Any

from app.core.logging import get_logger
from app.domain.enums import EvidenceType
from app.knowledge_graph.statements import find_gap_statements

logger = get_logger(__name__)

IRRELEVANT_BELOW = 0.45
COUNTER_RELEVANCE = 0.62
SUPPORT_RELEVANCE = 0.55

_SOLVES = re.compile(
    r"\b(we (propose|present|introduce|develop|design|build|demonstrate|show)|this (paper|study|work) (proposes|presents|introduces|develops)"
    r"|(a|an) (novel|new) (method|framework|approach|model|architecture|system|technique)|novel"
    r"|(achieve[sd]?|attain(s|ed)?|obtain(s|ed)?|outperform(s|ed)?|surpass(es|ed)?|improve[sd]?|enable[sd]?)\b"
    r"|state-of-the-art (performance|results))",
    re.IGNORECASE,
)
_WIDELY = re.compile(r"\b(has|have) been (widely|extensively|thoroughly|well|intensively) (studied|investigated|explored|addressed|researched)", re.IGNORECASE)


def _norm(v):
    s = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / s for x in v]


def _cos(a, b) -> float:
    return sum(x * y for x, y in zip(a, b))


class RuleEvidenceClassifier:
    """
    Parameters
    ----------
    embedder:
        ``embed_batch(texts)`` provider (the run's bge model).
    concept_forms:
        ``{concept name: [surface forms]}`` - the gap's concepts and their KG aliases.
    """

    def __init__(self, embedder: Any) -> None:
        self._embedder = embedder

    def classify_all(self, gap: dict, hits: list[dict], concept_forms: dict[str, list[str]]) -> list[dict]:
        """Classify every hit; returns the hits with evidence_type, relevance, coverage, rationale."""
        if not hits:
            return []
        texts = [gap["gap_statement"]] + [h["text"][:2000] for h in hits]
        vecs = [_norm(v) for v in self._embedder.embed_batch(texts)]
        gap_vec = vecs[0]
        patterns = {
            name: re.compile(r"(?<![\w-])(" + "|".join(re.escape(f) for f in sorted(set(forms), key=len, reverse=True)) + r")(?![\w-])", re.IGNORECASE)
            for name, forms in concept_forms.items() if forms
        }
        out = []
        for hit, vec in zip(hits, vecs[1:]):
            relevance = _cos(gap_vec, vec)
            text = hit["text"]
            named = [n for n, p in patterns.items() if p.search(text)]
            coverage = len(named) / len(patterns) if patterns else 0.0
            etype, why = self._decide(text, relevance, coverage, hit.get("section_type") or "ABSTRACT")
            out.append({**hit, "evidence_type": etype.value, "relevance": round(relevance, 4),
                        "concept_coverage": round(coverage, 4), "concepts_named": named, "rationale": why})
        return out

    @staticmethod
    def _decide(text: str, relevance: float, coverage: float, section_type: str) -> tuple[EvidenceType, str]:
        if relevance < IRRELEVANT_BELOW:
            return EvidenceType.INSUFFICIENT, f"low relevance to the gap ({relevance:.2f})"
        if _WIDELY.search(text) and coverage >= 0.5:
            return EvidenceType.CONTRADICTORY, "states the topic has already been widely studied"
        solves = _SOLVES.search(text)
        if solves and coverage >= 0.999 and relevance >= COUNTER_RELEVANCE:
            return EvidenceType.COUNTER, f"reports a solution ('{solves.group(0)}') covering all gap concepts, relevance {relevance:.2f}"
        if solves and (coverage >= 0.5 or relevance >= COUNTER_RELEVANCE):
            return EvidenceType.PARTIAL, f"reports a solution ('{solves.group(0)}') covering {coverage:.0%} of the gap concepts"
        states = find_gap_statements(text, section_type or "ABSTRACT")
        if states and (coverage >= 0.5 or relevance >= SUPPORT_RELEVANCE):
            return EvidenceType.SUPPORTING, f"states the limitation ('{states[0].cue}')"
        return EvidenceType.INSUFFICIENT, f"related (relevance {relevance:.2f}) but neither states nor fills the gap"


class LLMEvidenceClassifier:
    """
    Optional LLM classifier (one call per gap/item pair).  Not implemented: the
    rule + embedding classifier above is used so that validation is free,
    deterministic and reproducible.
    """

    def __init__(self, llm_provider: object) -> None:
        self._llm = llm_provider

    def classify(self, gap: Any, chunk: Any) -> EvidenceType:
        logger.warning("LLMEvidenceClassifier is not implemented; use RuleEvidenceClassifier.")
        return EvidenceType.INSUFFICIENT
