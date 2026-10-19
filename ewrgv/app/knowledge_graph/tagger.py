"""
app/knowledge_graph/tagger.py
-------------------------------
Dictionary + regex entity tagger (no model download, deterministic).

Sources, merged into one vocabulary per run:
  1. the seed lexicons in lexicon.py;
  2. the run's stored query understanding (methods, method_variants,
     datasets, metrics, task, entities, concepts, terminology_variants,
     constraints), typed by field;
  3. a dataset-name regex ("<Name> dataset", "<Name> database").

Matching is whole-word, longest surface form first, never overlapping.
Surface forms written with a leading "=" in the lexicon match
case-sensitively (short acronyms such as DR, FA, SVM).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from app.knowledge_graph.lexicon import ENTITY_TYPES

# Query-understanding field -> entity type
QU_FIELD_TYPES: dict[str, str] = {
    "methods": "METHOD",
    "method_variants": "METHOD",
    "datasets": "DATASET",
    "metrics": "METRIC",
    "entities": "DISEASE",
    "concepts": "CONCEPT",
    "terminology_variants": "CONCEPT",
    "constraints": "POPULATION",
}

# Words that look like dataset names in "<Name> dataset" but are not names.
_DATASET_STOP = frozenset({
    "this", "the", "our", "their", "each", "every", "a", "an", "public", "publicly", "large", "small",
    "same", "new", "proposed", "test", "training", "validation", "image", "images", "medical",
    "available", "open", "private", "benchmark", "whole", "entire", "full", "original", "used",
    "second", "first", "third", "different", "other", "two", "three", "both", "following",
    "clinical", "retinal", "fundus", "oct", "eye", "kaggle", "own", "custom", "combined",
})
_RE_DATASET = re.compile(r"\b([A-Z][A-Za-z0-9\-_]{2,}(?:[ -][A-Z0-9][A-Za-z0-9\-_]*)?)\s+(?:dataset|database|data set)s?\b")


@dataclass(frozen=True)
class Mention:
    entity_type: str
    canonical: str
    surface: str
    start: int
    end: int


def normalise_term(term: str) -> str:
    """Lower-case, unify hyphens/spaces and strip a plural 's' for comparison."""
    t = " ".join(term.lower().replace("-", " ").replace("_", " ").split())
    if len(t) > 4 and t.endswith("s") and not t.endswith("ss"):
        t = t[:-1]
    return t


class EntityTagger:
    """
    Tags entity mentions in text.

    Parameters
    ----------
    extra_terms:
        ``{entity_type: [term, ...]}`` added to the seed lexicons (e.g. from
        the query understanding).  Each term is its own canonical name unless
        it matches an existing surface form.
    """

    def __init__(self, extra_terms: Optional[dict[str, Iterable[str]]] = None) -> None:
        # surface key -> (type, canonical); case-insensitive and case-sensitive maps
        self._ci: dict[str, tuple[str, str]] = {}
        self._cs: dict[str, tuple[str, str]] = {}
        for etype, entries in ENTITY_TYPES.items():
            for canonical, forms in entries.items():
                for form in [canonical, *forms]:
                    self._add(etype, canonical, form)
        for etype, terms in (extra_terms or {}).items():
            for term in terms or []:
                term = (term or "").strip()
                if len(term) < 3 or len(term) > 60:
                    continue
                if term.lower() in self._ci:
                    continue  # already known, keep the lexicon's type/canonical
                self._add(etype, term.lower(), term)
        self._re_ci = self._compile(self._ci, re.IGNORECASE)
        self._re_cs = self._compile(self._cs, 0)

    def _add(self, etype: str, canonical: str, form: str) -> None:
        if form.startswith("="):
            self._cs.setdefault(form[1:], (etype, canonical))
        else:
            self._ci.setdefault(form.lower(), (etype, canonical))

    @staticmethod
    def _compile(table: dict[str, tuple[str, str]], flags: int) -> Optional[re.Pattern]:
        if not table:
            return None
        forms = sorted(table, key=len, reverse=True)
        alternation = "|".join(re.escape(f) for f in forms)
        return re.compile(rf"(?<![\w-])(?:{alternation})(?![\w-])", flags)

    @property
    def vocabulary_size(self) -> int:
        return len(self._ci) + len(self._cs)

    def tag(self, text: str) -> list[Mention]:
        """All non-overlapping mentions in ``text``, longest match first, in text order."""
        found: list[Mention] = []
        if self._re_ci is not None:
            for m in self._re_ci.finditer(text):
                etype, canonical = self._ci[m.group(0).lower()]
                found.append(Mention(etype, canonical, m.group(0), m.start(), m.end()))
        if self._re_cs is not None:
            for m in self._re_cs.finditer(text):
                etype, canonical = self._cs[m.group(0)]
                found.append(Mention(etype, canonical, m.group(0), m.start(), m.end()))
        for m in _RE_DATASET.finditer(text):
            name = m.group(1)
            if name.split()[0].lower() in _DATASET_STOP or name.lower() in self._ci:
                continue
            found.append(Mention("DATASET", name.lower(), name, m.start(1), m.end(1)))
        # Resolve overlaps: longer span wins, then earlier.
        found.sort(key=lambda x: (-(x.end - x.start), x.start))
        taken: list[tuple[int, int]] = []
        kept: list[Mention] = []
        for mention in found:
            if any(mention.start < e and s < mention.end for s, e in taken):
                continue
            taken.append((mention.start, mention.end))
            kept.append(mention)
        return sorted(kept, key=lambda x: x.start)


def query_terms(understanding: Optional[dict]) -> dict[str, list[str]]:
    """Typed terms from a stored query understanding (query_runs.understanding)."""
    out: dict[str, list[str]] = {}
    if not understanding:
        return out
    for field_name, etype in QU_FIELD_TYPES.items():
        values = understanding.get(field_name) or []
        if isinstance(values, str):
            values = [values]
        out.setdefault(etype, []).extend(v for v in values if isinstance(v, str))
    task = understanding.get("task")
    if isinstance(task, str) and task.strip():
        out.setdefault("TASK", []).append(task)
    return out
