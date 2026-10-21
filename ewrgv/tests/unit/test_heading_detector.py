"""
tests/unit/test_heading_detector.py
-------------------------------------
Unit tests for app/ingestion/sectioning/heading_detector.py

Tests cover all 10 detection signals and critical false-positive guards.
"""
import pytest
from app.ingestion.sectioning.heading_detector import detect_headings, HeadingCandidate


def make_lines(*texts: str, page: int = 1) -> list[tuple[int, int, str]]:
    """Helper to build a line list for the detector."""
    return [(i, page, t) for i, t in enumerate(texts)]


class TestHeadingDetectorBasicDetection:
    def test_known_vocab_unnumbered(self):
        lines = make_lines("", "Introduction", "")
        candidates = detect_headings(lines)
        assert any(c.raw_text == "Introduction" for c in candidates)

    def test_arabic_numbered(self):
        lines = make_lines("", "3. Methodology", "")
        candidates = detect_headings(lines)
        assert any(c.raw_text == "3. Methodology" for c in candidates)

    def test_arabic_numbered_subsection(self):
        lines = make_lines("", "3.1 Experimental Setup", "")
        candidates = detect_headings(lines)
        assert any(c.raw_text == "3.1 Experimental Setup" for c in candidates)

    def test_level_derived_from_numbering(self):
        lines = make_lines("", "3.1 Dataset", "")
        candidates = detect_headings(lines)
        match = next((c for c in candidates if "3.1" in c.raw_text), None)
        assert match is not None
        assert match.level == 2

    def test_all_caps_known_vocab(self):
        lines = make_lines("", "INTRODUCTION", "")
        candidates = detect_headings(lines)
        assert any(c.raw_text == "INTRODUCTION" for c in candidates)

    def test_conclusion_detected(self):
        lines = make_lines("", "Conclusion", "")
        candidates = detect_headings(lines)
        assert any(c.raw_text == "Conclusion" for c in candidates)


class TestHeadingDetectorFalsePositiveGuards:
    def test_figure_caption_not_heading(self):
        lines = make_lines("", "Figure 3. Proposed architecture.", "")
        candidates = detect_headings(lines)
        assert not any("Figure" in c.raw_text for c in candidates)

    def test_table_caption_not_heading(self):
        lines = make_lines("", "Table 2. Performance comparison.", "")
        candidates = detect_headings(lines)
        assert not any("Table" in c.raw_text for c in candidates)

    def test_page_number_not_heading(self):
        lines = make_lines("", "42", "")
        candidates = detect_headings(lines)
        assert not any(c.raw_text == "42" for c in candidates)

    def test_long_prose_line_not_heading(self):
        prose = ("This is a long prose sentence that should not be detected "
                 "as a section heading because it is clearly prose text.")
        lines = make_lines("", prose, "")
        candidates = detect_headings(lines)
        assert not any(c.raw_text == prose for c in candidates)

    def test_citation_line_not_heading(self):
        lines = make_lines("", "[1] Smith et al. 2022. Some paper title.", "")
        candidates = detect_headings(lines)
        assert len(candidates) == 0

    def test_too_long_line_not_heading(self):
        long_line = "A " * 70
        lines = make_lines("", long_line.strip(), "")
        candidates = detect_headings(lines)
        assert not any(c.raw_text == long_line.strip() for c in candidates)


class TestHeadingDetectorScores:
    def test_score_is_between_0_and_1(self):
        lines = make_lines("", "Introduction", "")
        candidates = detect_headings(lines)
        for c in candidates:
            assert 0.0 <= c.score <= 1.0

    def test_numbered_known_vocab_has_high_score(self):
        lines = make_lines("", "3. Methodology", "")
        candidates = detect_headings(lines)
        match = next((c for c in candidates if "Methodology" in c.raw_text), None)
        if match:
            assert match.score >= 0.7

    def test_signals_list_not_empty_for_accepted_heading(self):
        lines = make_lines("", "Introduction", "")
        candidates = detect_headings(lines)
        for c in candidates:
            assert len(c.signals) > 0


class TestHeadingDetectorMultiPage:
    def test_page_number_preserved(self):
        lines = [
            (0, 1, ""),
            (1, 1, "Introduction"),
            (2, 1, ""),
            (3, 2, ""),
            (4, 2, "3. Methodology"),
            (5, 2, ""),
        ]
        candidates = detect_headings(lines)
        intro = next((c for c in candidates if c.raw_text == "Introduction"), None)
        method = next((c for c in candidates if "Methodology" in c.raw_text), None)
        assert intro is not None and intro.page_number == 1
        assert method is not None and method.page_number == 2


class TestHeadingDetectorPdfNoise:
    """False headings seen on real PDFs (wrapped lines, references, page markers)."""

    REAL = ["1. Introduction", "A. Overview of the Proposed System", "4. Results",
            "Limitations", "CONCLUSION", "2.1 Dataset Description", "II. RELATED WORK"]
    NOISE = [
        "Zhang et al. (2014) proposed an automated",
        "retinal disease diagnosis system based on image",
        "27. Radosavovic, I, Kosaraju, R, Girshick, R, He, K, Dollar, P. Designing network",
        "20 / 21",
        "ISSN: 2502-4752",
        "Medical Image Processing, Texture Analysis,",
        "The proposed Local Binary Pattern (LBP)-based",
        "Binary Pattern-based retinal image analysis",
        "also demonstrated strong performance across",
        "larger clinical datasets to further improve",
    ]

    @staticmethod
    def _isolated(texts: list[str]) -> list[tuple[int, int, str]]:
        """Every line wrapped in blank lines, as in PDFs with a break after each phrase."""
        out: list[str] = []
        for t in texts:
            out += ["", t, ""]
        out.append("This is a long paragraph line with many words in it to follow headings.")
        return make_lines(*out)

    def test_noise_rejected_real_headings_kept(self) -> None:
        found = [h.raw_text for h in detect_headings(self._isolated(self.REAL + self.NOISE))]
        assert found == self.REAL

    @pytest.mark.parametrize("line", NOISE)
    def test_each_noise_line_rejected_alone(self, line: str) -> None:
        lines = make_lines("", line, "", "This is a long paragraph line with many words in it to follow.")
        assert detect_headings(lines) == []
