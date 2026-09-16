"""
tests/unit/test_enums.py
--------------------------
Unit tests for EWRGV domain enumerations.
"""

from __future__ import annotations

import pytest

from app.domain.enums import (
    EvidenceType,
    GapClassification,
    GapType,
    PipelineStage,
    QueryType,
)


class TestGapType:
    def test_all_values_present(self):
        expected = {
            "Methodological", "Population", "Dataset", "Geographic",
            "Temporal", "Contradictory Evidence", "Evaluation", "Other / Unknown",
        }
        assert {g.value for g in GapType} == expected

    def test_string_compatibility(self):
        assert GapType.METHODOLOGICAL == "Methodological"


class TestEvidenceType:
    def test_all_values_present(self):
        values = {e.value for e in EvidenceType}
        assert "Supporting" in values
        assert "Counter" in values
        assert "Partial" in values
        assert "Contradictory" in values
        assert "Insufficient" in values


class TestGapClassification:
    def test_three_outcomes(self):
        assert GapClassification.VALID_GAP == "VALID_GAP"
        assert GapClassification.UNCERTAIN_GAP == "UNCERTAIN_GAP"
        assert GapClassification.UNSUPPORTED_GAP == "UNSUPPORTED_GAP"


class TestPipelineStage:
    def test_ewrgv_stage_exists(self):
        assert PipelineStage.EWRGV_VALIDATION is not None

    def test_stage_ordering_exists(self):
        stages = list(PipelineStage)
        created_idx = stages.index(PipelineStage.CREATED)
        ewrgv_idx = stages.index(PipelineStage.EWRGV_VALIDATION)
        completed_idx = stages.index(PipelineStage.COMPLETED)
        assert created_idx < ewrgv_idx < completed_idx
