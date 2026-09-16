"""
tests/unit/test_domain_models.py
---------------------------------
Unit tests for EWRGV domain models.

Tests that:
- Models can be instantiated with valid data.
- Default values are correct.
- Domain logic properties work (e.g. ValidationResult.supporting_count).
"""

from __future__ import annotations

import pytest

from app.domain.enums import EvidenceType, GapClassification, GapType, QueryType
from app.domain.models.candidate_gap import CandidateGap, StructuredGapRepresentation
from app.domain.models.evidence import Evidence, Provenance
from app.domain.models.paper import Author, DocumentChunk, Paper
from app.domain.models.query import ResearchQuery
from app.domain.models.validation import CoverageAssessment, ValidationResult


class TestResearchQuery:
    def test_default_fields(self):
        q = ResearchQuery(query="What are the gaps in transformer models?")
        assert q.query == "What are the gaps in transformer models?"
        assert q.query_type == QueryType.UNKNOWN
        assert q.expanded_queries == []
        assert q.query_id is not None

    def test_with_type(self):
        q = ResearchQuery(query="test", query_type=QueryType.METHODOLOGICAL)
        assert q.query_type == QueryType.METHODOLOGICAL


class TestPaper:
    def test_default_fields(self):
        p = Paper(title="Attention Is All You Need")
        assert p.title == "Attention Is All You Need"
        assert p.authors == []
        assert p.abstract == ""
        assert p.paper_id is not None

    def test_with_authors(self):
        a = Author(name="Vaswani", affiliation="Google Brain")
        p = Paper(title="Test", authors=[a])
        assert p.authors[0].name == "Vaswani"


class TestDocumentChunk:
    def test_basic(self):
        chunk = DocumentChunk(paper_id="paper-1", text="Transformers are great.")
        assert chunk.paper_id == "paper-1"
        assert chunk.chunk_id is not None


class TestCandidateGap:
    def test_default_fields(self):
        gap = CandidateGap(gap_statement="No study addresses X in low-resource languages.")
        assert gap.primary_gap_type == GapType.OTHER
        assert gap.secondary_gap_types == []
        assert gap.initial_confidence == 0.0
        assert gap.gap_id is not None

    def test_with_type(self):
        gap = CandidateGap(
            gap_statement="test",
            primary_gap_type=GapType.METHODOLOGICAL,
        )
        assert gap.primary_gap_type == GapType.METHODOLOGICAL

    def test_structured_representation(self):
        rep = StructuredGapRepresentation(
            gap_statement="test gap",
            domain="NLP",
            key_concepts=["transformers", "low-resource"],
        )
        gap = CandidateGap(gap_statement="test", structured_representation=rep)
        assert gap.structured_representation.domain == "NLP"
        assert "transformers" in gap.structured_representation.key_concepts


class TestEvidence:
    def test_default_fields(self):
        e = Evidence(gap_id="gap-1", paper_id="paper-1", passage_id="chunk-1")
        assert e.evidence_type == EvidenceType.INSUFFICIENT
        assert e.relevance == 0.0
        assert e.strength == 0.0
        assert e.evidence_id is not None

    def test_provenance(self):
        p = Provenance(search_type="direct", query_used="transformers in NLP")
        e = Evidence(
            gap_id="g", paper_id="p", passage_id="c", provenance=p
        )
        assert e.provenance.search_type == "direct"


class TestCoverageAssessment:
    def test_all_none_by_default(self):
        c = CoverageAssessment()
        assert c.retrieval_coverage is None
        assert c.concept_coverage is None
        assert c.corpus_adequacy is None

    def test_partial_assignment(self):
        c = CoverageAssessment(retrieval_coverage=0.75, concept_coverage=0.6)
        assert c.retrieval_coverage == 0.75


class TestValidationResult:
    def test_default_fields(self):
        r = ValidationResult(gap_id="gap-1")
        assert r.gap_id == "gap-1"
        assert r.evidence == []
        assert r.classification is None
        assert not r.is_classified

    def test_supporting_counter_count(self):
        e1 = Evidence(
            gap_id="g", paper_id="p1", passage_id="c1",
            evidence_type=EvidenceType.SUPPORTING,
        )
        e2 = Evidence(
            gap_id="g", paper_id="p2", passage_id="c2",
            evidence_type=EvidenceType.COUNTER,
        )
        e3 = Evidence(
            gap_id="g", paper_id="p3", passage_id="c3",
            evidence_type=EvidenceType.SUPPORTING,
        )
        r = ValidationResult(gap_id="g", evidence=[e1, e2, e3])
        assert r.supporting_count == 2
        assert r.counter_count == 1

    def test_is_classified(self):
        r = ValidationResult(gap_id="g", classification=GapClassification.VALID_GAP)
        assert r.is_classified
