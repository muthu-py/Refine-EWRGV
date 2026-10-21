"""
app/domain/models/__init__.py
------------------------------
Exports all domain model classes for convenient single-location imports.

    from app.domain.models import Paper, PaperChunk, CandidateGap, ...
"""

from app.domain.models.paper import Author, Paper
from app.domain.models.paper_chunk import ChunkingResult, PaperChunk
from app.domain.models.query import ResearchQuery
from app.domain.models.query_understanding import QueryUnderstanding
from app.domain.models.candidate_evidence import (
    CandidateEvidenceAssessment,
    CandidateEvidenceItem,
)
from app.domain.models.candidate_gap import CandidateGap
from app.domain.models.evidence import Evidence
from app.domain.models.validation import CoverageAssessment, ValidationResult
from app.domain.models.collection import (
    PaperProvenance,
    CollectedPaper,
    CollectionResult,
)
from app.domain.models.retrieval_result import RetrievalResult

__all__ = [
    "Author",
    "Paper",
    "PaperChunk",
    "ChunkingResult",
    "ResearchQuery",
    "QueryUnderstanding",
    "CandidateGap",
    "CandidateEvidenceItem",
    "CandidateEvidenceAssessment",
    "Evidence",
    "CoverageAssessment",
    "ValidationResult",
    # Phase 2.2 — Literature Collection
    "PaperProvenance",
    "CollectedPaper",
    "CollectionResult",
    # Phase 2.5 — Retrieval
    "RetrievalResult",
]
