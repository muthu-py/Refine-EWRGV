"""
app/validation/validator.py
-----------------------------
EWRGVValidator - the EWRGV validation pipeline for one candidate gap.

    candidate gap (from POST /research/gaps)
          |
    generate_verification_queries()      query_generation.py (KG neighbourhood)
          |
    run_validation_searches()            direct / terminology / method_variation / counter
          |                              over run corpus + OpenAlex + Semantic Scholar
    classify_evidence()                  evidence/classifier.py (rules + embeddings)
          |
    aggregate_evidence()                 evidence/aggregator.py, evidence/strength.py
          |
    analyze_coverage()                   coverage/analyzer.py  (+ external coverage)
          |      \\__ iterate with refined queries while coverage is insufficient (iteration.py)
    score_confidence()                   scoring/confidence.py
          |
    classify_gap()                       scoring/classification.py (thresholds from settings)
          |
    generate_explanation()               explanation/evidence_chain.py, explanation/generator.py
          |
    ValidationResult (+ a readable dict for the API)
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Optional

from app.core.config import settings
from app.core.logging import get_logger
from app.coverage.analyzer import CoverageAnalyzer
from app.domain.models.candidate_gap import CandidateGap
from app.domain.models.validation import ValidationResult
from app.evidence.aggregator import EvidenceAggregator, to_evidence
from app.evidence.classifier import RuleEvidenceClassifier
from app.evidence.strength import strength_of
from app.explanation.evidence_chain import build_evidence_chain
from app.explanation.generator import TemplateExplanationGenerator
from app.scoring.classification import classify_by_confidence, reason_code
from app.scoring.confidence import EWRGVConfidenceScorer
from app.validation.counter_search import CounterEvidenceSearch
from app.validation.direct_search import ALL_SOURCES, DirectEvidenceSearch
from app.validation.iteration import ValidationIterationController
from app.validation.method_search import MethodVariationSearch
from app.validation.query_generation import VerificationQueries, VerificationQueryGenerator
from app.validation.terminology_search import TerminologySearch

logger = get_logger(__name__)


class EWRGVValidator:
    """
    Parameters
    ----------
    kg, corpus, embedder, retrieval:
        The run's knowledge graph, corpus, embedding provider and RetrievalService.
    external:
        ExternalLiterature (OpenAlex / Semantic Scholar); None = run corpus only.
    sources:
        Subset of ("corpus", "openalex", "semantic_scholar").
    per_query:
        Results requested per query per source.
    max_iterations:
        Defaults to settings.MAX_VALIDATION_ITERATIONS.
    """

    def __init__(self, kg: Any, corpus: Any, embedder: Any, retrieval: Any = None, external: Any = None,
                 sources: tuple[str, ...] = ALL_SOURCES, per_query: int = 10,
                 max_iterations: Optional[int] = None, anchor: Optional[str] = None) -> None:
        from app.gap_detection.signals.temporal import field_anchor

        self._kg = kg
        self._embedder = embedder
        self._anchor = anchor or field_anchor(kg)
        self._generator = VerificationQueryGenerator(kg, self._anchor)
        kwargs = dict(corpus=corpus, retrieval=retrieval, external=external if set(sources) - {"corpus"} else None,
                      sources=sources, per_query=per_query)
        self._searches = {
            "direct": DirectEvidenceSearch(**kwargs), "terminology": TerminologySearch(**kwargs),
            "method_variation": MethodVariationSearch(**kwargs), "counter": CounterEvidenceSearch(**kwargs),
        }
        self._classifier = RuleEvidenceClassifier(embedder)
        self._aggregator = EvidenceAggregator()
        self._coverage = CoverageAnalyzer(external if "openalex" in sources else None)
        self._scorer = EWRGVConfidenceScorer()
        self._explainer = TemplateExplanationGenerator()
        self._iteration = ValidationIterationController(max_iterations or settings.MAX_VALIDATION_ITERATIONS)

    @property
    def unavailable_sources(self) -> list[str]:
        return sorted(set().union(*(s.unavailable for s in self._searches.values())))

    # ------------------------------------------------------------------ #

    async def validate_candidate(self, gap: dict) -> tuple[ValidationResult, dict]:
        kg = self._kg
        concept_ids = [c["id"] for c in gap.get("concepts", []) if c["id"] in kg.entities]
        concept_forms = {kg.entities[e]["name"]: [kg.entities[e]["name"]] + [a for a in kg.entities[e].get("aliases", []) if len(a) > 2]
                         for e in concept_ids}
        if not concept_forms:
            from app.validation.query_generation import _keywords
            concept_forms = {k: [k] for k in _keywords(gap["gap_statement"], 3)}
        reference_year = (gap.get("signals", {}).get("explicit_statements", {}) or {}).get("latest_statement_year") \
            or max((kg.papers[p].get("year") or 0 for p in kg.on_topic_papers), default=None)
        run_papers = self._run_papers(concept_ids)

        queries = self._generator.generate(gap)
        all_queries: dict[str, list[str]] = {k: [] for k in queries.by_type()}
        hits: list[dict] = []
        used: set[str] = set()
        iteration, items, coverage = 0, [], None
        while True:
            iteration += 1
            for stype, qs in queries.by_type().items():
                fresh = [q for q in qs if q not in used]
                used.update(fresh)
                all_queries[stype] += fresh
                for q in fresh:
                    hits += await self._searches[stype].run_query(q)
            classified = await asyncio.to_thread(self._classifier.classify_all, gap, hits, concept_forms)
            for h in classified:
                h["strength"] = strength_of(h["relevance"], h.get("year"), reference_year, h["source"])
            items = self._aggregator.aggregate_items(classified)
            coverage = self._coverage.analyze(list(concept_forms), items, all_queries, run_papers)
            relevant = sum(1 for it in items if it["evidence_type"] != "Insufficient")
            if not self._iteration.should_iterate(iteration, coverage, relevant):
                break
            queries = self._iteration.refine_queries(self._generator, gap, used)
            if not any(queries.by_type().values()):
                break

        external = await self._coverage.external_coverage(list(concept_forms)[:4] if concept_ids else [], self._anchor)
        coverage = self._coverage.analyze(list(concept_forms), items, all_queries, run_papers, external)
        scoring = self._scorer.score(gap["gap_score"], items, coverage, external)
        classification = classify_by_confidence(scoring["confidence"], settings.CONFIDENCE_THRESHOLD_VALID,
                                                settings.CONFIDENCE_THRESHOLD_UNCERTAIN)
        reason = reason_code(classification, scoring["components"], gap["detectors"])
        cov_dict = coverage.model_dump(exclude={"notes"})
        chain = build_evidence_chain(gap, all_queries, items, cov_dict, external, scoring, classification.value, reason, iteration)
        explanation = self._explainer.generate(gap, classification.value, reason, scoring["confidence"], chain)

        relevant_items = [it for it in items if it["evidence_type"] != "Insufficient"]
        result = ValidationResult(
            gap_id=gap["gap_id"],
            evidence=[to_evidence(gap["gap_id"], it, reference_year) for it in relevant_items[:40]],
            coverage=coverage, confidence=scoring["confidence"], classification=classification,
            explanation=explanation, evidence_chain=chain, validation_iterations=iteration,
            metadata={"reason": reason, "external_coverage": external, "scoring": scoring["components"],
                      "anchor": self._anchor},
        )
        view = {
            "gap_id": gap["gap_id"], "gap_rank": gap.get("rank"), "gap_statement": gap["gap_statement"],
            "gap_type": gap["gap_type"], "detectors": gap["detectors"], "gap_score": gap["gap_score"],
            "researchrag_score": gap.get("researchrag_score"),
            "classification": classification.value, "confidence": scoring["confidence"], "reason": reason,
            "explanation": explanation,
            "coverage": {**cov_dict, "notes": coverage.notes},
            "external_coverage": external,
            "scoring": scoring["components"],
            "iterations": iteration,
            "queries": all_queries,
            "evidence_counts": scoring["components"]["evidence_counts"],
            "evidence": [self._evidence_view(it) for it in relevant_items[:25]],
            "evidence_chain": [e.model_dump() for e in chain],
            "unavailable_sources": self.unavailable_sources,
        }
        return result, view

    def _run_papers(self, concept_ids: list[str]) -> int:
        if not concept_ids:
            return 0
        sets = [set(self._kg.entities[e]["papers"]) for e in concept_ids]
        both = set.intersection(*sets) if len(sets) > 1 else sets[0]
        return len(both) if len(sets) > 1 and both else len(set.union(*sets))

    @staticmethod
    def _evidence_view(it: dict) -> dict:
        text = " ".join(it["text"].split())
        return {
            "evidence_type": it["evidence_type"], "strength": it["strength"], "relevance": it["relevance"],
            "source": it["source"], "title": it.get("title"), "year": it.get("year"), "doi": it.get("doi"),
            "url": it.get("url"), "section": it.get("section"), "section_type": it.get("section_type"),
            "pages": it.get("pages"), "concepts_named": it.get("concepts_named"), "why": it["rationale"],
            "found_by": sorted({f"{f['search_type']}: {f['query']}" for f in it.get("found_by", [])})[:4],
            "snippet": text[:350] + ("..." if len(text) > 350 else ""),
        }

    # ------------------------------------------------------------------ #
    # Domain-model entry points (CandidateGap) kept for the pipeline interface
    # ------------------------------------------------------------------ #

    def validate(self, gap: CandidateGap) -> ValidationResult:
        rep = gap.structured_representation
        concepts = []
        for name in (rep.key_concepts if rep else []):
            eid = next((e for e, v in self._kg.entities.items() if v["name"].lower() == name.lower()), None)
            if eid:
                concepts.append({"id": eid, "name": name})
        as_dict = {"gap_id": gap.gap_id, "gap_statement": gap.gap_statement, "gap_type": gap.primary_gap_type.value,
                   "detectors": [gap.metadata.get("detector", "llm")], "gap_score": gap.evidence_assessment.score if gap.evidence_assessment else 0.0,
                   "concepts": concepts, "signals": {}, "score_notes": {"stating_papers": len(gap.source_papers)}}
        return asyncio.run(self.validate_candidate(as_dict))[0]

    def validate_batch(self, gaps: list[CandidateGap]) -> list[ValidationResult]:
        return [self.validate(g) for g in gaps]
