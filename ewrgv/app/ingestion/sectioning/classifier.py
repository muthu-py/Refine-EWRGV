"""
app/ingestion/sectioning/classifier.py
----------------------------------------
Rule-based semantic section type classifier for Phase 2.6.

Design notes
------------
* Deterministic keyword/pattern matching — no LLM calls.
* Uses the normalized_title produced by normalizer.py.
* Maps academic heading variants to SectionType enum values.
* When no confident match is found, returns SectionType.OTHER.
* The original heading title is NEVER modified here.

Classification operates on normalized_title (lower-cased internally).
"""

from __future__ import annotations

import re
from app.domain.enums import SectionType


# ---------------------------------------------------------------------------
# Classification rules: ordered list of (pattern, SectionType)
# Earlier rules take priority. All patterns are matched on lower-cased normalized title.
# ---------------------------------------------------------------------------

_RULES: list[tuple[re.Pattern, SectionType]] = [
    # ----- Abstract -----
    (re.compile(r"\babstract\b"), SectionType.ABSTRACT),

    # ----- References / Bibliography -----
    (re.compile(r"\b(references|bibliography|literature cited|works cited|reference list)\b"),
     SectionType.REFERENCES),

    # ----- Acknowledgments -----
    (re.compile(r"\backnowledg(e?ments?|ements?)\b"), SectionType.ACKNOWLEDGMENTS),

    # ----- Appendix -----
    (re.compile(r"\b(appendix|appendices|supplementary material|supplemental material|additional (results|experiments|details))\b"),
     SectionType.APPENDIX),

    # ----- Introduction -----
    (re.compile(r"\b(introduction|overview|motivation)\b"), SectionType.INTRODUCTION),

    # ----- Related Work (before Background to catch combined forms) -----
    (re.compile(r"\b(related work|related studies|related literature|related research|prior work|previous work)\b"),
     SectionType.RELATED_WORK),
    (re.compile(r"\b(background and related|background & related|related and background)\b"),
     SectionType.RELATED_WORK),

    # ----- Literature Review -----
    (re.compile(r"\b(literature review|literature survey|systematic review|scoping review)\b"),
     SectionType.LITERATURE_REVIEW),

    # ----- Background -----
    (re.compile(r"\b(background|preliminar(y|ies)|foundations|basics|notation|definitions|problem (statement|formulation))\b"),
     SectionType.BACKGROUND),

    # ----- Dataset / Data -----
    (re.compile(r"\b(dataset|datasets|data collection|data sources|data prep|corpus|data description)\b"),
     SectionType.DATASET),
    (re.compile(r"^data$"), SectionType.DATA),
    (re.compile(r"\b(data)\b"), SectionType.DATA),

    # ----- Experimental Setup -----
    (re.compile(r"\b(experimental (setup|settings?|protocol|evaluation)|evaluation (setup|protocol|settings?)|setup and (implementation|evaluation)|baselines?|configurations?)\b"),
     SectionType.EXPERIMENTAL_SETUP),

    # ----- Experiments -----
    (re.compile(r"\b(experiments?|ablation (study|studies))\b"), SectionType.EXPERIMENTS),

    # ----- Evaluation -----
    (re.compile(r"\b(evaluation|performance evaluation|benchmark(ing)?|comparison|comparative (study|evaluation))\b"),
     SectionType.EVALUATION),

    # ----- Results & Analysis -----
    # Combined form MUST come BEFORE standalone RESULTS rule
    (re.compile(r"\b(results? and discussion|results? & discussion|discussion and results?)\b"),
     SectionType.DISCUSSION),

    (re.compile(r"\b(results?|findings|quantitative results?|qualitative results?|performance|case study)\b"),
     SectionType.RESULTS),

    # ----- Discussion -----
    (re.compile(r"\b(discussion|analysis)\b"), SectionType.DISCUSSION),

    # ----- Limitations -----
    (re.compile(r"\b(limitations?( of (the|this) (study|work|paper|approach))?|study limitations?)\b"),
     SectionType.LIMITATIONS),

    # ----- Future Work (combined "limitations and future work" → FUTURE_WORK) -----
    (re.compile(r"\b(future (work|directions?|research)|research (directions?|agenda)|limitations? and future)\b"),
     SectionType.FUTURE_WORK),

    # ----- Conclusion -----
    (re.compile(r"\b(conclusions?|concluding remarks?|final remarks?|summary( and conclusions?)?)\b"),
     SectionType.CONCLUSION),

    # ----- Methodology -----
    (re.compile(r"\b(methodology|proposed (method|approach|framework|system|model)|our (approach|method|model|framework)|technical approach|approach|algorithm)\b"),
     SectionType.METHODOLOGY),

    # ----- Methods -----
    (re.compile(r"\b(methods?|materials? and methods?|materials? & methods?)\b"),
     SectionType.METHODS),

    # ----- Model -----
    (re.compile(r"\b(model( architecture)?|network architecture|neural network|transformer|architecture)\b"),
     SectionType.MODEL),

    # ----- System Design -----
    (re.compile(r"\b(system (design|architecture|overview|description)|design)\b"),
     SectionType.SYSTEM_DESIGN),

    # ----- Implementation -----
    (re.compile(r"\b(implementation( details?)?|technical details?)\b"),
     SectionType.IMPLEMENTATION),
]


def classify(normalized_title: str) -> SectionType:
    """
    Classify a normalized section heading into a SectionType.

    Parameters
    ----------
    normalized_title:
        The heading text with numbering stripped and title-cased
        (output of normalizer.normalize_heading).

    Returns
    -------
    SectionType:
        Best matching type, or SectionType.OTHER if no confident match.
    """
    lower = normalized_title.strip().lower()
    if not lower:
        return SectionType.UNKNOWN

    for pattern, section_type in _RULES:
        if pattern.search(lower):
            return section_type

    return SectionType.OTHER
