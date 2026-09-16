"""
app/domain/models/__init__.py
------------------------------
Exports all domain model classes for convenient single-location imports.

    from app.domain.models import Paper, DocumentChunk, CandidateGap, ...
"""

from app.domain.models.paper import Author, Paper, DocumentChunk
from app.domain.models.query import ResearchQuery
from app.domain.models.query_understanding import QueryUnderstanding
from app.domain.models.candidate_gap import CandidateGap
from app.domain.models.evidence import Evidence
from app.domain.models.validation import CoverageAssessment, ValidationResult
from app.domain.models.collection import (
    PaperProvenance,
    CollectedPaper,
    CollectionResult,
)

__all__ = [
    "Author",
    "Paper",
    "DocumentChunk",
    "ResearchQuery",
    "QueryUnderstanding",
    "CandidateGap",
    "Evidence",
    "CoverageAssessment",
    "ValidationResult",
    # Phase 2.2 — Literature Collection
    "PaperProvenance",
    "CollectedPaper",
    "CollectionResult",
]
