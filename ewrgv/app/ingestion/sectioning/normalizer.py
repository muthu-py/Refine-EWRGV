"""
app/ingestion/sectioning/normalizer.py
----------------------------------------
Heading normalization for Phase 2.6 — Section Extraction.

Normalization strips superficial presentation differences while
ALWAYS preserving the original title in the ParsedSection.title field.

The normalized_title is used for:
- Section type classification
- Deduplication of heading variants

Examples
--------
"3. Introduction"     -> "Introduction"
"INTRODUCTION"        -> "Introduction"
"1.2 Related Work"    -> "Related Work"
"II. METHODOLOGY"     -> "Methodology"
"A. Appendix"         -> "Appendix"
"3.1.2 Data Collection" -> "Data Collection"
"""

from __future__ import annotations

import re

# Compiled patterns for prefix stripping
_RE_ARABIC_PREFIX = re.compile(r"^\d{1,3}(?:\.\d{1,3}){0,4}\.?\s+")
_RE_ROMAN_PREFIX = re.compile(
    r"^(?:I{1,4}|IV|VI{0,4}|IX|X{1,4}|XL|L|XC|C)\.?\s+",
    re.IGNORECASE,
)
_RE_LETTER_PREFIX = re.compile(r"^[A-Z]\.?\s+")

# Words that should remain lower-case in title-case output
_LOWER_WORDS = frozenset({
    "a", "an", "the", "and", "but", "or", "nor", "for", "so", "yet",
    "as", "at", "by", "in", "of", "on", "to", "up", "via", "with",
    "&",
})


def normalize_heading(raw: str) -> str:
    """
    Normalize a heading by removing numbering prefixes and applying
    consistent capitalization.

    Parameters
    ----------
    raw:
        Original heading text (may include numbering like "3.1 Dataset").

    Returns
    -------
    str:
        Normalized heading with numbering stripped and title-cased.
    """
    text = raw.strip()

    # Step 1: Strip known numbering prefixes
    text = _strip_prefix(text)

    # Step 2: Apply consistent capitalization
    text = _title_case(text)

    return text.strip()


def _strip_prefix(text: str) -> str:
    """Remove leading numbering prefix from heading text."""
    # Try arabic first (most common)
    result = _RE_ARABIC_PREFIX.sub("", text, count=1)
    if result != text:
        return result

    # Try roman numerals
    result = _RE_ROMAN_PREFIX.sub("", text, count=1)
    if result != text:
        return result

    # Try single uppercase letter prefix
    result = _RE_LETTER_PREFIX.sub("", text, count=1)
    if result != text:
        return result

    return text


def _title_case(text: str) -> str:
    """
    Apply title-case to the text.

    All-caps words are converted to title-case.
    Short connector words (and, or, of, etc.) are lower-cased unless first.

    Examples
    --------
    "INTRODUCTION"        -> "Introduction"
    "RELATED WORK"        -> "Related Work"
    "results and discussion" -> "Results and Discussion"
    """
    if not text:
        return text

    words = text.split()
    result = []
    for i, word in enumerate(words):
        # Strip trailing punctuation for comparison
        clean = word.rstrip(".,:;")
        punct = word[len(clean):]

        lower_clean = clean.lower()

        if i == 0:
            # Always capitalize first word
            result.append(clean.capitalize() + punct)
        elif lower_clean in _LOWER_WORDS and i < len(words) - 1:
            result.append(lower_clean + punct)
        elif clean.isupper() and len(clean) > 1:
            # ALL-CAPS word → Title Case
            result.append(clean.capitalize() + punct)
        else:
            # Already mixed or title case — capitalize if first char is letter
            if clean and clean[0].isalpha():
                result.append(clean[0].upper() + clean[1:] + punct)
            else:
                result.append(word)

    return " ".join(result)
