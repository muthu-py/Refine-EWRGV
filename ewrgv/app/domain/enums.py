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
