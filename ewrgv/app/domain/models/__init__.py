"""
app/domain/models/__init__.py
------------------------------
Exports all domain model classes for convenient single-location imports.

    from app.domain.models import Paper, DocumentChunk, CandidateGap, ...
"""

from app.domain.models.paper import Author, Paper, DocumentChunk
from app.domain.models.query import ResearchQuery
from app.domain.models.candidate_gap import CandidateGap
from app.domain.models.evidence import Evidence
from app.domain.models.validation import CoverageAssessment, ValidationResult

__all__ = [
    "Author",
    "Paper",
    "DocumentChunk",
    "ResearchQuery",
    "CandidateGap",
    "Evidence",
    "CoverageAssessment",
    "ValidationResult",
]
