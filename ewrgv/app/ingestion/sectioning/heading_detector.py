"""
app/ingestion/sectioning/heading_detector.py
---------------------------------------------
Context-aware, multi-signal heading detection for academic papers.

Design rationale
----------------
A heading is NOT detected by a single condition like "short line = heading".
That produces excessive false positives.

Instead, each candidate line is scored across multiple signals.
A candidate that scores above the acceptance threshold is accepted as a heading.

Signals
-------
1.  Numbering pattern   – arabic (1, 1., 1.2, 1.2.3), roman (I., II.), letter (A., A)
2.  Known vocabulary    – matches academic heading vocabulary list
3.  Line length         – headings are typically < 80 characters
4.  Capitalization      – ALL-CAPS, Title Case, or sentence-case starting a known word
5.  Not a caption       – "Figure", "Fig.", "Table", "Algorithm" at the start
6.  Not a numbered list – line ends with period/colon AND is part of list context
7.  Not a page number   – purely numeric or very short pure-digit line
8.  Not an equation     – contains math symbols that override heading interpretation
9.  Not a sentence      – line that ends in a period with many words is likely prose
10. Standalone line     – line is preceded and followed by whitespace/empty lines
    (checked via surrounding context)

Heading candidates are returned as HeadingCandidate dataclasses.
The SectionExtractor uses these to determine section boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class HeadingCandidate:
    """A potential section heading detected in the document."""
    line_index: int          # 0-based index into the page-concatenated line list
    page_number: int         # 1-based page number where this heading appears
    raw_text: str            # Original text of the line
    score: float             # Detection confidence [0.0, 1.0]
    level: int               # Inferred hierarchy depth (1 = top, 2 = sub, etc.)
    numbering: str           # Detected prefix (e.g. "3.1", "I", "A") or ""
    signals: list[str] = field(default_factory=list)   # Diagnostic list


# ---------------------------------------------------------------------------
# Scoring thresholds
# ---------------------------------------------------------------------------

# Minimum score for a line to be accepted as a heading
_ACCEPTANCE_THRESHOLD = 0.45

# Maximum accepted heading text length (characters)
_MAX_HEADING_LENGTH = 120

# ---------------------------------------------------------------------------
# Compiled regular expressions
# ---------------------------------------------------------------------------

# Arabic numbered headings: "1", "1.", "1.2", "1.2.", "1.2.3", "10.2.4.1"
_RE_ARABIC = re.compile(
    r"^(\d{1,3}(?:\.\d{1,3}){0,4}\.?)\s+(.+)$"
)

# Roman numeral headings: "I.", "II.", "III.", "I ", "II " etc.
_RE_ROMAN = re.compile(
    r"^((?:I{1,4}|IV|VI{0,4}|IX|X{1,4}|XL|L|XC|C)\.?)\s+(.+)$",
    re.IGNORECASE,
)

# Lettered headings: "A.", "B.", "A ", "B " etc. — upper-case single letter
_RE_LETTER = re.compile(
    r"^([A-Z]\.?)\s+(.+)$"
)

# Figure / table captions — should NOT be treated as headings
_RE_CAPTION = re.compile(
    r"^(?:fig(?:ure)?\.?|table|algorithm|listing|exhibit|plate|appendix\s+[a-z])\s*\d",
    re.IGNORECASE,
)

# Page number lines — purely numeric, maybe with whitespace
_RE_PAGE_NUMBER = re.compile(r"^\s*\d{1,4}\s*$")

# Math equation indicators
_RE_MATH = re.compile(r"[=+\-*/^{}\\|∑∫∂≤≥≈×÷]")

# Sentence-like ending (many words + period) — likely prose, not heading
_RE_SENTENCE_END = re.compile(r"\w{3,}\s+\w{3,}\s+\w{3,}.*\.$")

# Citation line patterns — e.g. "[1] Smith et al..."
_RE_CITATION = re.compile(r"^\s*\[\d+\]|\s*\(\d{4}\)$")

# ALL-CAPS line
_RE_ALL_CAPS = re.compile(r"^[A-Z][A-Z\s\d:&/-]{2,}$")

# Title-case line (most words capitalized)
def _is_title_case(text: str) -> bool:
    words = text.split()
    if len(words) < 1:
        return False
    cap_words = sum(
        1 for w in words
        if w and w[0].isupper() and not w.isupper()  # capitalized but not ALL-CAPS word
    )
    return cap_words / len(words) >= 0.6 and len(words) <= 12


# ---------------------------------------------------------------------------
# Exhaustive academic heading vocabulary
# ---------------------------------------------------------------------------

# All entries are lower-cased for comparison.  Partial matches are OK.
_KNOWN_HEADINGS: frozenset[str] = frozenset({
    # Abstract / intro
    "abstract", "introduction", "overview", "motivation", "preface",
    # Background
    "background", "preliminaries", "preliminary", "basics", "foundations",
    "notation", "definitions", "problem statement", "problem formulation",
    # Related work
    "related work", "related studies", "related literature", "related research",
    "prior work", "previous work", "literature review", "literature survey",
    "survey", "background and related work", "background & related work",
    # Methods
    "methodology", "method", "methods", "proposed method", "proposed approach",
    "approach", "framework", "proposed framework", "system overview",
    "system description", "architecture", "materials and methods",
    "materials & methods", "technical approach", "algorithm",
    # Data
    "dataset", "datasets", "data", "data collection", "data sources",
    "data preprocessing", "preprocessing", "data preparation",
    "corpus", "data description",
    # Model / system
    "model", "model architecture", "network architecture", "system design",
    "system architecture", "implementation", "implementation details",
    "technical details", "design",
    # Experiments
    "experiments", "experiment", "experimental setup", "experimental settings",
    "setup", "settings", "configuration", "baseline", "baselines",
    "evaluation setup", "evaluation protocol",
    # Results
    "results", "findings", "results and discussion", "results & discussion",
    "analysis", "quantitative results", "qualitative results",
    "performance", "ablation study", "ablation", "case study",
    # Evaluation
    "evaluation", "performance evaluation", "benchmark",
    # Discussion
    "discussion", "discussion and conclusion", "discussion and conclusions",
    # Limitations
    "limitations", "limitations of the study", "study limitations",
    "limitations and future work", "limitations & future work",
    "limitations and future directions",
    # Future work
    "future work", "future directions", "future research", "research directions",
    "research agenda", "conclusion and future work", "conclusions and future work",
    "limitations and future directions", "limitations & future directions",
    # Conclusion
    "conclusion", "conclusions", "concluding remarks", "concluding remarks",
    "final remarks", "summary", "summary and conclusion", "summary and conclusions",
    "summary & conclusions", "conclusion and future work",
    # References
    "references", "bibliography", "literature cited", "works cited",
    "reference list",
    # Acknowledgments
    "acknowledgments", "acknowledgements", "acknowledgment", "acknowledgement",
    # Appendix
    "appendix", "appendices", "supplementary material", "supplementary materials",
    "supplemental material", "additional results", "additional experiments",
    "additional details",
    # Threat model / security papers
    "threat model", "security analysis", "attack model",
    # Ethics / impact
    "ethics", "ethical considerations", "broader impact", "social impact",
    "reproducibility",
    # Additional common patterns
    "related", "our approach", "our method", "our model", "proposed",
})


# ---------------------------------------------------------------------------
# Core detector
# ---------------------------------------------------------------------------


def detect_headings(
    lines: list[tuple[int, int, str]],  # (line_index, page_number, text)
    min_score: float = _ACCEPTANCE_THRESHOLD,
) -> list[HeadingCandidate]:
    """
    Detect heading candidates from a list of (line_index, page_number, text) tuples.

    Parameters
    ----------
    lines:
        All lines from the document concatenated across pages, preserving page
        boundaries via the page_number field.
    min_score:
        Minimum detection score to accept a candidate.

    Returns
    -------
    list[HeadingCandidate]:
        Headings in document order.
    """
    candidates: list[HeadingCandidate] = []
    n = len(lines)

    for i, (line_idx, page_num, raw) in enumerate(lines):
        stripped = raw.strip()
        if not stripped:
            continue

        # Pre-filters — reject obvious non-headings early
        if _RE_PAGE_NUMBER.match(stripped):
            continue
        if _RE_CITATION.match(stripped):
            continue
        if len(stripped) > _MAX_HEADING_LENGTH:
            continue

        score, signals, level, numbering = _score_line(
            stripped=stripped,
            index=i,
            lines=lines,
        )

        if score >= min_score:
            candidates.append(HeadingCandidate(
                line_index=line_idx,
                page_number=page_num,
                raw_text=stripped,
                score=score,
                level=level,
                numbering=numbering,
                signals=signals,
            ))

    return candidates


# ---------------------------------------------------------------------------
# Scoring implementation
# ---------------------------------------------------------------------------


def _score_line(
    stripped: str,
    index: int,
    lines: list[tuple[int, int, str]],
) -> tuple[float, list[str], int, str]:
    """
    Score a single candidate line.

    Returns (score, signals, level, numbering_prefix).
    """
    score = 0.0
    signals: list[str] = []
    level = 1
    numbering = ""

    lower = stripped.lower()
    n = len(lines)

    # --- Signal 1: Caption guard (strong negative) ---
    if _RE_CAPTION.match(stripped):
        return 0.0, ["CAPTION_GUARD"], 1, ""

    # --- Signal 2: Math guard ---
    if _RE_MATH.search(stripped) and not _is_known_vocab(lower):
        math_ratio = sum(1 for c in stripped if c in "=+*^{}\\|∑∫∂") / max(len(stripped), 1)
        if math_ratio > 0.05:
            return 0.0, ["MATH_GUARD"], 1, ""

    # --- Signal 3: Sentence guard ---
    if _RE_SENTENCE_END.match(stripped) and len(stripped.split()) > 7:
        return 0.0, ["SENTENCE_GUARD"], 1, ""

    # --- Signal 4: Numbered heading pattern ---
    arabic_m = _RE_ARABIC.match(stripped)
    roman_m = _RE_ROMAN.match(stripped)
    letter_m = _RE_LETTER.match(stripped)

    if arabic_m:
        num_part = arabic_m.group(1).rstrip(".")
        rest = arabic_m.group(2).strip()
        dots = num_part.count(".")
        level = dots + 1
        numbering = num_part

        # Validate: rest must not be a typical numbered list sentence
        if not _looks_like_list_item(rest):
            score += 0.5
            signals.append(f"ARABIC_NUMBERED(L{level})")

    elif roman_m:
        rest = roman_m.group(2).strip()
        numbering = roman_m.group(1).rstrip(".")
        level = 1
        if not _looks_like_list_item(rest):
            score += 0.4
            signals.append("ROMAN_NUMBERED")

    elif letter_m:
        rest = letter_m.group(2).strip()
        numbering = letter_m.group(1).rstrip(".")
        # Letter headings need additional support to avoid false positives
        # (e.g. "A note on..." should not trigger)
        if _is_known_vocab(rest.lower()) or _is_all_caps_or_title(rest):
            score += 0.35
            signals.append("LETTER_HEADED")
        else:
            score += 0.1  # weak signal only
    else:
        rest = stripped

    # --- Signal 5: Known academic vocabulary ---
    if _is_known_vocab(lower):
        score += 0.5
        signals.append("KNOWN_VOCAB")
    else:
        # Partial vocabulary match — substring in known headings
        for known in _KNOWN_HEADINGS:
            if known in lower and len(known) > 4:
                score += 0.25
                signals.append(f"PARTIAL_VOCAB({known})")
                break

    # --- Signal 6: Length heuristic ---
    word_count = len(stripped.split())
    if 1 <= word_count <= 8:
        score += 0.15
        signals.append("SHORT_LINE")
    elif word_count <= 12:
        score += 0.05
        signals.append("MEDIUM_LINE")

    # --- Signal 7: Capitalization ---
    if _RE_ALL_CAPS.match(stripped) and word_count <= 6:
        score += 0.2
        signals.append("ALL_CAPS")
    elif _is_title_case(stripped):
        score += 0.1
        signals.append("TITLE_CASE")

    # --- Signal 8: Surrounding whitespace context ---
    prev_empty = _prev_is_empty(index, lines)
    next_empty = _next_is_empty(index, lines)
    if prev_empty and next_empty:
        score += 0.2
        signals.append("ISOLATED_LINE")
    elif prev_empty or next_empty:
        score += 0.08
        signals.append("SEMI_ISOLATED")

    # --- Signal 9: Next line looks like paragraph content ---
    if _next_is_paragraph(index, lines):
        score += 0.1
        signals.append("FOLLOWED_BY_PARAGRAPH")

    # --- Signal 10: Numbered list guard ---
    # If the "heading" ends with a period and the next lines also start with numbers,
    # it's likely a list item, not a heading.
    if arabic_m and stripped.endswith(".") and _next_also_numbered(index, lines):
        score -= 0.35
        signals.append("NUMBERED_LIST_GUARD")

    return min(score, 1.0), signals, level, numbering


# ---------------------------------------------------------------------------
# Helper predicates
# ---------------------------------------------------------------------------


def _is_known_vocab(lower_text: str) -> bool:
    """Check if the text (stripped of numbering) matches known vocabulary."""
    stripped_lower = _strip_numbering(lower_text)
    return stripped_lower in _KNOWN_HEADINGS


def _strip_numbering(text: str) -> str:
    """Remove leading numbering prefix for vocabulary comparison."""
    text = text.strip()
    # Try arabic
    m = _RE_ARABIC.match(text)
    if m:
        return m.group(2).strip().lower()
    # Try roman
    m = _RE_ROMAN.match(text)
    if m:
        return m.group(2).strip().lower()
    # Try letter
    m = _RE_LETTER.match(text)
    if m:
        return m.group(2).strip().lower()
    return text.lower()


def _is_all_caps_or_title(text: str) -> bool:
    return bool(_RE_ALL_CAPS.match(text)) or _is_title_case(text)


def _looks_like_list_item(text: str) -> bool:
    """
    Return True if the text after a numbering prefix looks like a prose
    list item rather than a section title.

    Heuristics:
    - Ends with a period (prose sentence)
    - Starts with a lowercase word (prose continuation)
    - Contains more than 12 words (too long for a heading)
    """
    words = text.split()
    if len(words) > 12:
        return True
    if text.endswith(".") and len(words) > 4:
        return True
    if words and words[0][0].islower():
        return True
    return False


def _prev_is_empty(i: int, lines: list[tuple[int, int, str]]) -> bool:
    if i == 0:
        return True
    return not lines[i - 1][2].strip()


def _next_is_empty(i: int, lines: list[tuple[int, int, str]]) -> bool:
    if i >= len(lines) - 1:
        return True
    return not lines[i + 1][2].strip()


def _next_is_paragraph(i: int, lines: list[tuple[int, int, str]]) -> bool:
    """Return True if the next non-empty line looks like paragraph prose."""
    for j in range(i + 1, min(i + 4, len(lines))):
        candidate = lines[j][2].strip()
        if candidate:
            words = candidate.split()
            return len(words) >= 6
    return False


def _next_also_numbered(i: int, lines: list[tuple[int, int, str]]) -> bool:
    """Return True if the following non-empty lines start with arabic numbers too."""
    count = 0
    for j in range(i + 1, min(i + 6, len(lines))):
        candidate = lines[j][2].strip()
        if candidate and _RE_ARABIC.match(candidate):
            count += 1
    return count >= 2
