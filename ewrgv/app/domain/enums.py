"""
app/domain/enums.py
-------------------
Shared enumerations used across the EWRGV domain.

All enums are defined here (single source of truth) to prevent circular
imports and to make the taxonomy immediately visible at the package boundary.
"""

from __future__ import annotations

from enum import Enum


# ------------------------------------------------------------------ #
# Research Gap Taxonomy
# ------------------------------------------------------------------ #


class GapType(str, Enum):
    """
    Standard taxonomy for classifying the *nature* of a research gap.

    A CandidateGap carries exactly one primary GapType and may carry
    zero-or-more secondary GapTypes.
    """
    METHODOLOGICAL = "Methodological"
    POPULATION = "Population"
    DATASET = "Dataset"
    GEOGRAPHIC = "Geographic"
    TEMPORAL = "Temporal"
    CONTRADICTORY_EVIDENCE = "Contradictory Evidence"
    EVALUATION = "Evaluation"
    OTHER = "Other / Unknown"


# ------------------------------------------------------------------ #
# Evidence Classification
# ------------------------------------------------------------------ #


class EvidenceType(str, Enum):
    """
    Classification assigned to a single evidence item after the EWRGV
    evidence-classification step.

    A single paper may yield multiple evidence items with *different*
    EvidenceTypes (e.g. supporting for one claim, counter for another).
    """
    SUPPORTING = "Supporting"           # Confirms the gap exists
    COUNTER = "Counter"                 # Suggests the gap is already addressed
    PARTIAL = "Partial"                 # Partially addresses the gap
    CONTRADICTORY = "Contradictory"     # Directly contradicts the gap claim
    INSUFFICIENT = "Insufficient"       # Related but too weak to classify


# ------------------------------------------------------------------ #
# Validation Classification
# ------------------------------------------------------------------ #


class GapClassification(str, Enum):
    """
    Final classification produced by the EWRGV validator for each
    CandidateGap after the full validation pipeline completes.
    """
    VALID_GAP = "VALID_GAP"
    UNCERTAIN_GAP = "UNCERTAIN_GAP"
    UNSUPPORTED_GAP = "UNSUPPORTED_GAP"


# ------------------------------------------------------------------ #
# Query
# ------------------------------------------------------------------ #


class QueryType(str, Enum):
    """
    High-level classification of the research question posed by the user.
    Used to guide query expansion strategy.
    """
    BROAD_SURVEY = "Broad Survey"
    SPECIFIC_PROBLEM = "Specific Problem"
    COMPARATIVE = "Comparative"
    METHODOLOGICAL = "Methodological"
    APPLICATION = "Application"
    UNKNOWN = "Unknown"


# ------------------------------------------------------------------ #
# Pipeline Stage
# ------------------------------------------------------------------ #


class PipelineStage(str, Enum):
    """
    Tracks which stage of the research pipeline a job has reached.
    Used by the orchestration layer for status reporting.
    """
    CREATED = "created"
    QUERY_EXPANSION = "query_expansion"
    LITERATURE_COLLECTION = "literature_collection"
    DOCUMENT_PROCESSING = "document_processing"
    RETRIEVAL = "retrieval"
    GAP_DETECTION = "gap_detection"
    CANDIDATE_RANKING = "candidate_ranking"
    EWRGV_VALIDATION = "ewrgv_validation"
    EVIDENCE_AGGREGATION = "evidence_aggregation"
    CONFIDENCE_ASSESSMENT = "confidence_assessment"
    CLASSIFICATION = "classification"
    EXPLANATION = "explanation"
    COMPLETED = "completed"
    FAILED = "failed"


# ------------------------------------------------------------------ #
# Section Extraction (Phase 2.6)
# ------------------------------------------------------------------ #


class SectionType(str, Enum):
    """
    Controlled vocabulary for semantic section types in academic papers.

    Design notes
    ------------
    * NOT every paper will contain all section types. Missing sections are normal.
    * Combined headings (e.g. "Results and Discussion") use the closest type.
    * Original title is always preserved. This enum classifies, never renames.
    """
    ABSTRACT = "ABSTRACT"
    INTRODUCTION = "INTRODUCTION"
    BACKGROUND = "BACKGROUND"
    RELATED_WORK = "RELATED_WORK"
    LITERATURE_REVIEW = "LITERATURE_REVIEW"
    METHODOLOGY = "METHODOLOGY"
    METHODS = "METHODS"
    DATASET = "DATASET"
    DATA = "DATA"
    MODEL = "MODEL"
    SYSTEM_DESIGN = "SYSTEM_DESIGN"
    IMPLEMENTATION = "IMPLEMENTATION"
    EXPERIMENTS = "EXPERIMENTS"
    EXPERIMENTAL_SETUP = "EXPERIMENTAL_SETUP"
    RESULTS = "RESULTS"
    EVALUATION = "EVALUATION"
    DISCUSSION = "DISCUSSION"
    LIMITATIONS = "LIMITATIONS"
    FUTURE_WORK = "FUTURE_WORK"
    CONCLUSION = "CONCLUSION"
    REFERENCES = "REFERENCES"
    ACKNOWLEDGMENTS = "ACKNOWLEDGMENTS"
    APPENDIX = "APPENDIX"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


class SectionStatus(str, Enum):
    """
    Pipeline status for the section-extraction step (Phase 2.6).

    Kept separate from full_text_status (Phase 2.4) and parsing_status (Phase 2.5).
    """
    NOT_EXTRACTED = "NOT_EXTRACTED"
    EXTRACTED = "EXTRACTED"
    EXTRACTION_FAILED = "EXTRACTION_FAILED"


class ChunkStatus(str, Enum):
    """
    Pipeline status for the chunking step (Phase 2.7).

    Kept separate from parsing_status (Phase 2.5) and section_status (Phase 2.6).
    """
    NOT_CHUNKED = "NOT_CHUNKED"
    CHUNKED = "CHUNKED"
    CHUNKING_FAILED = "CHUNKING_FAILED"


class EmbeddingStatus(str, Enum):
    """
    Pipeline status for the embedding step (Phase 2.9).

    Tracks the embedding state of a paper's chunks.  A paper may be
    PARTIAL when some (but not all) chunks have been successfully embedded.
    """
    NOT_EMBEDDED = "NOT_EMBEDDED"
    EMBEDDING_IN_PROGRESS = "EMBEDDING_IN_PROGRESS"
    EMBEDDED = "EMBEDDED"
    EMBEDDING_FAILED = "EMBEDDING_FAILED"
    PARTIAL = "PARTIAL"

