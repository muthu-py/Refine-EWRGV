"""
app/coverage/analyzer.py
--------------------------
Coverage analyzer: computes a multi-dimensional CoverageAssessment for a
gap's evidence, plus external (bibliometric) coverage.

Dimensions
----------
retrieval_coverage    -> retrieval.py    share of queries with relevant hits
concept_coverage      -> concepts.py     share of concepts named in relevant evidence
terminology_coverage  -> terminology.py  share of alias queries with relevant hits
method_coverage       -> methods.py      share of sibling-method queries with relevant hits
corpus_adequacy       -> corpus.py       enough run + external literature behind the judgement

External coverage (OpenAlex hit counts, within the field anchor):

    cov_ext = n(C1 AND ... AND Ck AND field) / min_i n(Ci AND field)

For two concepts this is n(A AND B) / min(n(A), n(B)): the share of the
smaller concept's literature that already combines it with the other.  A gap
that is sparse in the run corpus but dense here is a corpus artefact.
Undefined (None) for single-concept gaps.
"""

from __future__ import annotations

from typing import Any, Optional

from app.core.logging import get_logger
from app.coverage.concepts import compute_concept_coverage
from app.coverage.corpus import compute_corpus_adequacy, compute_corpus_uncertainty
from app.coverage.methods import compute_method_coverage
from app.coverage.retrieval import compute_retrieval_coverage
from app.coverage.terminology import compute_terminology_coverage
from app.domain.models.validation import CoverageAssessment
from app.validation.sources import phrase_query

logger = get_logger(__name__)


class CoverageAnalyzer:
    def __init__(self, external: Any = None) -> None:
        self._external = external

    async def external_coverage(self, concepts: list[str], anchor: Optional[str]) -> Optional[dict]:
        terms = [c for c in concepts if c][:4]
        if self._external is None or len(terms) < 2:
            return None
        field = [anchor] if anchor and not any(anchor.lower() in t.lower() for t in terms) else []
        singles = {}
        for t in terms:
            n = await self._external.openalex_count(phrase_query([t] + field))
            if n is None:
                return {"available": False}
            singles[t] = n
        joint = await self._external.openalex_count(phrase_query(terms + field))
        if joint is None:
            return {"available": False}
        smallest = min(singles.values())
        return {
            "available": True, "terms": terms, "field": field[0] if field else None,
            "counts": singles, "joint": joint,
            "coverage": round(joint / max(1, smallest), 4),
            "formula": "n(all concepts AND field) / min_i n(concept_i AND field)",
        }

    def analyze(self, concepts: list[str], items: list[dict], queries: dict[str, list[str]],
                run_papers: int, external_coverage: Optional[dict] = None) -> CoverageAssessment:
        all_queries = [q for qs in queries.values() for q in qs]
        relevant_external = sum(1 for it in items if it["source"] != "corpus" and it["evidence_type"] != "Insufficient")
        adequacy = compute_corpus_adequacy(run_papers, relevant_external)
        notes = [
            f"{sum(1 for it in items if it['evidence_type'] != 'Insufficient')} relevant of {len(items)} distinct items",
            f"{run_papers} run papers mention the concepts; {relevant_external} relevant external items",
        ]
        if external_coverage and external_coverage.get("available"):
            notes.append(f"external coverage {external_coverage['coverage']:.3f} "
                         f"({external_coverage['joint']} joint / min {min(external_coverage['counts'].values())})")
        elif external_coverage is not None:
            notes.append("external coverage unavailable (OpenAlex did not answer)")
        return CoverageAssessment(
            retrieval_coverage=compute_retrieval_coverage(items, all_queries),
            concept_coverage=compute_concept_coverage(concepts, items),
            terminology_coverage=compute_terminology_coverage(items, queries.get("terminology", [])),
            method_coverage=compute_method_coverage(items, queries.get("method_variation", [])),
            corpus_adequacy=adequacy,
            corpus_uncertainty=compute_corpus_uncertainty(adequacy),
            notes=notes,
        )
