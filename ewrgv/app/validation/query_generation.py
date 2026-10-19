"""
app/validation/query_generation.py
------------------------------------
Generates EWRGV verification queries for a candidate gap from its concepts
and their knowledge-graph neighbourhood (no LLM needed).

    direct           the gap's concepts + the field anchor          "attention mechanism recurrent retinal"
    terminology      each concept replaced by a KG alias / synonym   "self-attention LSTM retinal"
    method_variation a METHOD / DATASET / MODALITY concept replaced
                     by a sibling of the same type that co-occurs
                     with the other concepts                         "transformer recurrent retinal"
    counter          phrased as if the gap were already solved       "attention mechanism recurrent retinal novel framework"

Each query is plain text (used for the run corpus, OpenAlex relevance
search and Semantic Scholar).  ``concept_terms`` are the canonical names
used for exact phrase counts in coverage analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)

_STOP = frozenset("""a an the and or of in on for to with by from as at via is are was were be been that this these those
there their its it which such than more most many several some other existing current models model methods method approaches
approach studies study research work works papers paper often still remains remain limited lack lacking need needs future""".split())
_COUNTER_SUFFIXES = ["novel framework", "proposed method improves"]


@dataclass
class VerificationQueries:
    """Typed container for queries generated for each EWRGV search type."""
    direct: list[str] = field(default_factory=list)
    terminology: list[str] = field(default_factory=list)
    method_variation: list[str] = field(default_factory=list)
    counter: list[str] = field(default_factory=list)
    concept_terms: list[str] = field(default_factory=list)
    anchor: Optional[str] = None

    def by_type(self) -> dict[str, list[str]]:
        return {"direct": self.direct, "terminology": self.terminology,
                "method_variation": self.method_variation, "counter": self.counter}

    def all(self) -> list[tuple[str, str]]:
        return [(t, q) for t, qs in self.by_type().items() for q in qs]


def _keywords(statement: str, k: int = 6) -> list[str]:
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", statement) if w.lower() not in _STOP]
    seen, out = set(), []
    for w in words:
        if w.lower() not in seen:
            seen.add(w.lower())
            out.append(w)
    return out[:k]


class VerificationQueryGenerator:
    """
    Parameters
    ----------
    kg:
        The run's KnowledgeGraph (aliases, co-occurrence neighbours).
    anchor:
        Field term added to every query to stay in the domain (e.g. "retinal").
    """

    def __init__(self, kg: Any, anchor: Optional[str] = None, llm_provider: object = None) -> None:
        self._kg = kg
        self._anchor = anchor
        self._llm = llm_provider  # unused: queries are built from the graph

    def generate(self, gap: dict) -> VerificationQueries:
        kg = self._kg
        concept_ids = [c["id"] for c in gap.get("concepts", []) if c["id"] in kg.entities]
        names = [kg.entities[e]["name"] for e in concept_ids]
        anchor = self._anchor
        tail = [anchor] if anchor and not any(anchor.lower() in n.lower() for n in names) else []
        if not names:
            names = _keywords(gap["gap_statement"], 5)
        base = " ".join(names + tail)
        q = VerificationQueries(concept_terms=[kg.entities[e]["name"] for e in concept_ids], anchor=anchor)
        q.direct = [base]
        # direct phrasing of the statement itself (key words), when it adds something
        kw = " ".join(_keywords(gap["gap_statement"], 7) + tail)
        if kw.lower() != base.lower():
            q.direct.append(kw)

        # terminology: aliases (prefer multi-word, different normalised form)
        for i, eid in enumerate(concept_ids):
            aliases = [a for a in kg.entities[eid].get("aliases", [])
                       if a.lower() != names[i].lower() and len(a) > 2 and a.lower() not in {n.lower() for n in names}]
            aliases.sort(key=lambda a: (-len(a.split()), -len(a)))
            seen = set()
            for alias in aliases:
                key = re.sub(r"[^a-z]", "", alias.lower())
                if key in seen or key == re.sub(r"[^a-z]", "", names[i].lower()):
                    continue
                seen.add(key)
                q.terminology.append(" ".join(names[:i] + [alias] + names[i + 1:] + tail))
                if len(seen) >= 2:
                    break
        q.terminology = list(dict.fromkeys(q.terminology))[:3]

        # method variation: same-type siblings co-occurring with the other concepts
        for i, eid in enumerate(concept_ids):
            e = kg.entities[eid]
            if e["type"] not in ("METHOD", "DATASET", "MODALITY"):
                continue
            for sib in self._siblings(eid, [c for c in concept_ids if c != eid])[:2]:
                q.method_variation.append(" ".join(names[:i] + [kg.entities[sib]["name"]] + names[i + 1:] + tail))
        q.method_variation = list(dict.fromkeys(q.method_variation))[:3]

        q.counter = [f"{base} {suffix}" for suffix in _COUNTER_SUFFIXES]
        return q

    def refine(self, gap: dict, used: set[str]) -> VerificationQueries:
        """Next-iteration queries: broader (no anchor, single concepts) and unused aliases."""
        first = self.generate(gap)
        kg = self._kg
        names = first.concept_terms or _keywords(gap["gap_statement"], 4)
        broader = [" ".join(names)] + [n for n in names if len(names) > 1]
        extra = VerificationQueries(concept_terms=first.concept_terms, anchor=first.anchor)
        extra.direct = [x for x in broader if x not in used][:2]
        for c in gap.get("concepts", []):
            for alias in kg.entities.get(c["id"], {}).get("aliases", [])[2:6]:
                cand = alias if not first.anchor else f"{alias} {first.anchor}"
                if cand not in used:
                    extra.terminology.append(cand)
        extra.terminology = extra.terminology[:2]
        extra.counter = [x for x in (f"{' '.join(names)} solved", f"{' '.join(names)} state of the art") if x not in used][:1]
        return extra

    def _siblings(self, eid: str, others: list[str]) -> list[str]:
        kg = self._kg
        etype = kg.entities[eid]["type"]
        neighbours: dict[str, int] = {}
        for edge in kg.cooccurrence:
            for a, b in ((edge["a"], edge["b"]), (edge["b"], edge["a"])):
                if a in others or (not others and a == eid):
                    if b != eid and kg.entities.get(b, {}).get("type") == etype and not kg.entities[b].get("generic"):
                        neighbours[b] = neighbours.get(b, 0) + edge["papers"]
        return [e for e, _ in sorted(neighbours.items(), key=lambda kv: -kv[1])]
