"""Rule-based extraction of method/task result claims for the run KG."""

from __future__ import annotations

import re
from typing import Any

from app.knowledge_graph.statements import split_sentences
from app.knowledge_graph.tagger import EntityTagger

_POSITIVE = re.compile(
    r"\b(outperform(?:s|ed)?|improv(?:e|es|ed|ement)|higher (?:accuracy|performance|sensitivity|specificity|auc|f1)|"
    r"better than|achiev(?:e|es|ed)\s+(?:an?\s+)?(?:accuracy|performance|score|auc|f1)|"
    r"increase[sd]?\s+(?:the\s+)?(?:accuracy|performance|sensitivity|specificity|auc|f1))\b", re.I)
_NEGATIVE = re.compile(
    r"\b(fail(?:s|ed)?|worse than|underperform(?:s|ed)?|no significant (?:difference|improvement)|"
    r"does not improve|did not improve|not improve|lower (?:accuracy|performance|sensitivity|specificity|auc|f1)|"
    r"decrease[sd]?\s+(?:the\s+)?(?:accuracy|performance|sensitivity|specificity|auc|f1))\b", re.I)
_METRIC_VALUE = re.compile(
    r"\b(accuracy|auc|auroc|f1(?:-score)?|sensitivity|specificity|precision|recall|dice(?: score)?|"
    r"kappa|iou)\b\s*(?:was|of|=|:)?\s*(\d+(?:\.\d+)?\s*%?)", re.I)
_RESULT_SECTIONS = {"RESULTS", "DISCUSSION", "ABSTRACT"}


def extract_result_claims(text: str, section_type: str, tagger: EntityTagger) -> list[dict[str, Any]]:
    """Extract evidence-backed directional claims from eligible prose sections."""
    if (section_type or "").upper() not in _RESULT_SECTIONS:
        return []
    claims: list[dict[str, Any]] = []
    seen: set[str] = set()
    for sentence in split_sentences(text):
        normalised = " ".join(sentence.lower().split())
        if normalised in seen:
            continue
        seen.add(normalised)
        mentions = tagger.tag(sentence)
        entity_ids = sorted({f"{m.entity_type}:{m.canonical}" for m in mentions})
        types = {m.entity_type for m in mentions}
        if "METHOD" not in types or not types.intersection({"DATASET", "TASK"}):
            continue
        positive, negative = bool(_POSITIVE.search(sentence)), bool(_NEGATIVE.search(sentence))
        if not positive and not negative:
            continue
        direction = "MIXED" if positive and negative else "POSITIVE" if positive else "NEGATIVE"
        metric = _METRIC_VALUE.search(sentence)
        claims.append({
            "sentence": sentence,
            "direction": direction,
            "metric": {"name": metric.group(1).lower(), "value": metric.group(2).strip()} if metric else None,
            "entities": entity_ids,
        })
    return claims
