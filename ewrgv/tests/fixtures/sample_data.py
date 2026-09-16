"""
tests/fixtures/sample_data.py
-------------------------------
Sample domain objects for use in tests.

All fixtures here create real domain model instances with valid data.
No external services are called.
"""

from __future__ import annotations

from app.domain.enums import EvidenceType, GapType
from app.domain.models.candidate_gap import CandidateGap, StructuredGapRepresentation
from app.domain.models.evidence import Evidence, Provenance
from app.domain.models.paper import Author, DocumentChunk, Paper
from app.domain.models.query import ResearchQuery


def make_paper(
    title: str = "Attention Is All You Need",
    paper_id: str = "paper-001",
) -> Paper:
    return Paper(
        paper_id=paper_id,
        title=title,
        authors=[Author(name="Vaswani", affiliation="Google Brain")],
        abstract="We propose a new architecture called the Transformer...",
        source="semantic_scholar",
    )


def make_chunk(
    paper_id: str = "paper-001",
    text: str = "The Transformer relies entirely on attention mechanisms.",
    section: str = "Introduction",
) -> DocumentChunk:
    return DocumentChunk(
        paper_id=paper_id,
        text=text,
        section=section,
    )


def make_query(query: str = "What are the gaps in Transformer-based models?") -> ResearchQuery:
    return ResearchQuery(query=query)


def make_candidate_gap(
    gap_statement: str = "No study evaluates Transformer models in low-resource settings.",
    gap_type: GapType = GapType.POPULATION,
) -> CandidateGap:
    rep = StructuredGapRepresentation(
        gap_statement=gap_statement,
        domain="NLP",
        population="Low-resource languages",
        key_concepts=["Transformer", "low-resource", "evaluation"],
    )
    return CandidateGap(
        gap_statement=gap_statement,
        primary_gap_type=gap_type,
        structured_representation=rep,
        source_papers=["paper-001"],
        initial_confidence=0.65,
    )


def make_evidence(
    gap_id: str = "gap-001",
    paper_id: str = "paper-001",
    evidence_type: EvidenceType = EvidenceType.SUPPORTING,
) -> Evidence:
    return Evidence(
        gap_id=gap_id,
        paper_id=paper_id,
        passage_id="chunk-001",
        evidence_type=evidence_type,
        passage_text="Transformers have not been evaluated in low-resource settings.",
        relevance=0.85,
        strength=0.70,
        provenance=Provenance(
            search_type="direct",
            query_used="Transformer low-resource evaluation",
            retrieval_rank=1,
        ),
    )
