"""
tests/fixtures/sample_data.py
-------------------------------
Sample domain objects for use in tests.

All fixtures here create real domain model instances with valid data.
No external services are called.
"""

from __future__ import annotations

from typing import Optional

from app.domain.enums import EvidenceType, GapType, SectionType
from app.domain.models.candidate_gap import CandidateGap, StructuredGapRepresentation
from app.domain.models.evidence import Evidence, Provenance
from app.domain.models.paper import Author, Paper
from app.domain.models.paper_chunk import PaperChunk
from app.domain.models.query import ResearchQuery
from app.ingestion.sectioning.classifier import classify


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
    chunk_id: Optional[str] = None,
    section_type: Optional[SectionType] = None,
    **kwargs,
) -> PaperChunk:
    """A PaperChunk whose section_type is classified from ``section`` like the real pipeline."""
    if section_type is None:
        section_type = classify(section) if section else SectionType.UNKNOWN
    fields = dict(
        paper_id=paper_id,
        section_id=f"sec-{paper_id}-{section or 'none'}",
        section_title=section,
        section_type=section_type,
        text=text,
        **kwargs,
    )
    if chunk_id is not None:
        fields["chunk_id"] = chunk_id
    return PaperChunk(**fields)


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
