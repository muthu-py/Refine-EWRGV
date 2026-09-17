"""
tests/unit/test_section_normalizer.py
--------------------------------------
Unit tests for app/ingestion/sectioning/normalizer.py
"""
import pytest
from app.ingestion.sectioning.normalizer import normalize_heading


class TestNormalizer:
    def test_arabic_numbered_l1(self):
        assert normalize_heading("3. Introduction") == "Introduction"

    def test_arabic_numbered_l2(self):
        assert normalize_heading("3.1 Experimental Setup") == "Experimental Setup"

    def test_arabic_numbered_l3(self):
        assert normalize_heading("3.1.2 Data Collection") == "Data Collection"

    def test_roman_numbered(self):
        result = normalize_heading("II. METHODOLOGY")
        assert result == "Methodology"

    def test_letter_headed(self):
        result = normalize_heading("A. Appendix")
        assert result == "Appendix"

    def test_all_caps_unnumbered(self):
        result = normalize_heading("INTRODUCTION")
        assert result == "Introduction"

    def test_all_caps_multiword(self):
        result = normalize_heading("RELATED WORK")
        assert result == "Related Work"

    def test_already_title_case(self):
        result = normalize_heading("Related Work")
        assert result == "Related Work"

    def test_mixed_case_preserved_properly(self):
        result = normalize_heading("results and discussion")
        assert result == "Results and Discussion"

    def test_empty_string(self):
        assert normalize_heading("") == ""

    def test_strips_trailing_dot_in_prefix(self):
        # "1." prefix should be removed
        assert normalize_heading("1. Background") == "Background"

    def test_connector_words_lowercase(self):
        result = normalize_heading("Methods and Materials")
        assert "and" in result.lower()

    def test_no_prefix_unchanged_capitalization(self):
        result = normalize_heading("Methods")
        assert result == "Methods"
