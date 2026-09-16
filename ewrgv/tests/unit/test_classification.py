"""
tests/unit/test_classification.py
-----------------------------------
Unit tests for the gap classification function in scoring/classification.py.

Tests the pure threshold logic without any infrastructure.
"""

from __future__ import annotations

import pytest

from app.domain.enums import GapClassification
from app.scoring.classification import classify_by_confidence


class TestClassifyByConfidence:
    """Tests for the pure confidence → classification mapping."""

    def test_high_confidence_is_valid(self):
        result = classify_by_confidence(0.80)
        assert result == GapClassification.VALID_GAP

    def test_exactly_at_valid_threshold(self):
        result = classify_by_confidence(0.70)
        assert result == GapClassification.VALID_GAP

    def test_mid_confidence_is_uncertain(self):
        result = classify_by_confidence(0.55)
        assert result == GapClassification.UNCERTAIN_GAP

    def test_exactly_at_uncertain_threshold(self):
        result = classify_by_confidence(0.40)
        assert result == GapClassification.UNCERTAIN_GAP

    def test_low_confidence_is_unsupported(self):
        result = classify_by_confidence(0.20)
        assert result == GapClassification.UNSUPPORTED_GAP

    def test_zero_confidence_is_unsupported(self):
        result = classify_by_confidence(0.0)
        assert result == GapClassification.UNSUPPORTED_GAP

    def test_custom_thresholds(self):
        result = classify_by_confidence(
            0.60,
            threshold_valid=0.80,
            threshold_uncertain=0.50,
        )
        assert result == GapClassification.UNCERTAIN_GAP

    def test_full_confidence_is_valid(self):
        result = classify_by_confidence(1.0)
        assert result == GapClassification.VALID_GAP
