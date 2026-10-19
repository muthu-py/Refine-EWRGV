"""
app/knowledge_graph/statements.py
-----------------------------------
Explicit gap-statement extraction (the automatic version of SciSpace /
Elicit "Limitations" and "Future work" columns).

A gap statement is a sentence, from a section where authors discuss
limitations or open problems, that contains a gap cue.  Each statement gets
a cue class and a strength:

    OPEN_PROBLEM  "remains unexplored", "has not been studied", "few studies"   1.0
    LIMITATION    "limited to", "only evaluated on", "lack of", "struggle with"  0.8
    FUTURE_WORK   "future work should", "remains to be", "could be explored"    0.6

Statements about the paper's own product (UI, app, "our system") or plain
contribution claims are dropped: they do not describe a gap in the field.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Sections in which authors state limitations / open problems.  ABSTRACT and
# INTRODUCTION are included at reduced weight: many papers state the problem
# they address there ("however, X remains limited").
SECTION_WEIGHTS: dict[str, float] = {
    "LIMITATIONS": 1.0,
    "FUTURE_WORK": 1.0,
    "CONCLUSION": 0.9,
    "DISCUSSION": 0.9,
    "RESULTS": 0.6,
    "LITERATURE_REVIEW": 0.7,
    "RELATED_WORK": 0.7,
    "BACKGROUND": 0.6,
    "INTRODUCTION": 0.6,
    "ABSTRACT": 0.6,
    "OTHER": 0.5,
    "UNKNOWN": 0.5,
}
# Section headings that mark gap discussion regardless of section_type.
_GAP_HEADING = re.compile(
    r"research gap|limitation|challenge|open (problem|issue|question)|future (work|direction|research)|discussion|conclusion",
    re.IGNORECASE,
)

_CUES: list[tuple[str, float, re.Pattern]] = [
    ("OPEN_PROBLEM", 1.0, re.compile(
        r"\b(remain(s|ed)?\s+(largely\s+)?(unexplored|unaddressed|unresolved|unclear|understudied|under-explored|an open|a (major|key|significant|critical) challenge|challenging)"
        r"|(has|have)\s+(not|rarely|seldom)\s+been\s+(studied|explored|investigated|addressed|examined|evaluated|considered)"
        r"|(few|limited|no)\s+(prior\s+)?(studies|works|research|attempts)\s+(have\s+)?(explored|addressed|investigated|examined|focus|focused|considered)"
        r"|(is|are)\s+(still\s+)?(lacking|missing|scarce|rare)"
        r"|open (problem|issue|question|challenge)s?"
        r"|(several|many|important|critical|key)\s+challenges\s+(remain|persist|are yet)"
        r"|yet to be (overcome|addressed|explored|resolved)"
        r"|research gap)",
        re.IGNORECASE)),
    ("LIMITATION", 0.8, re.compile(
        r"\b(limited (to|by|in|generali[sz]ation|ability|capability|applicability)|is limited|are limited|remains limited|limitations? (of|in|include)"
        r"|only (tested|evaluated|validated|trained|applied|considered)\s+on|on a single|single[- ](dataset|centre|center|modality|disease|type)"
        r"|lack(s|ing)?\s+of|lack\s+(the|a|an|sufficient|adequate)|insufficient|inadequate"
        r"|(struggle|fail)(s|ed)?\s+(to|with)|suffer(s|ed)?\s+from|(is|are)\s+(sensitive|prone)\s+to"
        r"|(depend(s|ence|ency)?|reliance|rel(y|ies) heavily)\s+on\s+(large|labell?ed|annotated|manual|expert)"
        r"|restricts?\s+(their|its)\s+applicability|tend to focus on|focus (mainly|primarily|only) on"
        r"|(does|do|did) not (account|consider|address|generali[sz]e|capture|handle)|overlook(s|ed|ing)?"
        r"|(poor|reduced|degraded|low) (generali[sz]ation|performance|robustness|interpretability))",
        re.IGNORECASE)),
    ("FUTURE_WORK", 0.6, re.compile(
        r"\b(future (work|research|studies|directions?|efforts?)\s+(should|could|will|may|might|need|needs|must|is|are|can|include|includes|on|to|in)"
        r"|in (the )?future,? (we|it|research|studies|work)|remains? to be (seen|explored|investigated|studied|determined)"
        r"|(could|should|need to|needs to|must) be (further )?(explored|investigated|studied|addressed|extended|validated|evaluated)"
        r"|further (research|work|studies|investigation|validation) (is|are) (needed|required|necessary|warranted)"
        r"|warrants? further|calls? for|there is a need|need for (more|further|larger|robust|better))",
        re.IGNORECASE)),
]

# Sentences describing the paper's own product / contribution, not a field gap.
_SELF_REFERENCE = re.compile(
    r"\b(this (app|application|tool|website|interface|project|chatbot)|our (app|application|tool|website|interface|chatbot)"
    r"|user[- ]friendly|user interface|graphical interface|the proposed (system|application) (offers|provides|allows)"
    r"|we (propose|present|introduce|develop|developed|proposed) (a|an|the)\b(?!.*\b(lack|limited|fail|challenge)))",
    re.IGNORECASE,
)

# Contribution claims: the paper says it has *solved* something.  Such sentences
# mention limitations ("to overcome the limitation of X, we ...") but they
# describe what is addressed, not what is missing.
_CONTRIBUTION = re.compile(
    r"\b(the proposed (system|method|model|framework|approach|architecture|technique|pipeline)"
    r"|our (proposed )?(method|model|framework|approach|system|architecture|study|work)"
    r"|(we|this (study|paper|work|research|project|review)) (propose[sd]?|present(s|ed)?|implement(s|ed)?|develop(s|ed)?"
    r"|introduce[sd]?|overcome[s]?|address(es|ed)?|tackle[sd]?|demonstrate[sd]?|achieve[sd]?)"
    r"|to (overcome|address|tackle|mitigate) (the|these|this|such|those)"
    r"|overcomes? (the|these) limitations?|(advantages|benefits) over"
    # progress claims: "advances in X resolve / mitigate the limitation"
    r"|(resolve[sd]?|alleviate[sd]?|mitigate[sd]?|solve[sd]?|eliminate[sd]?)\b)",
    re.IGNORECASE,
)
# Motivation sentences about pre-automation practice ("Traditional examination
# relies on experts...") describe why automation is needed, not a research gap.
_MOTIVATION = re.compile(r"^\s*(traditional|conventional|manual|current clinical|existing clinical)\b", re.IGNORECASE)
# Flattened comparison tables: "Limitation of the paper 1 ...", repeated metric
# cells ("Accuracy 96.4% ... Accuracy 94%"), bullet runs, "Follow-ups:" lists.
_TABLE_LIKE = re.compile(
    r"\blimitations? of the paper\b|\bfollow-ups?:|(\b(accuracy|precision|recall|f1)\b[^.]{0,40}?\d+(\.\d+)?\s*%.*){2,}"
    r"|(•[^•]{0,200}){3,}",
    re.IGNORECASE,
)
# Numbered title-case rows ("1 Computer-Assisted Diagnosis For ... 2 Deep Convolutional ...") - case-sensitive.
_TABLE_ROWS = re.compile(r"(\b\d{1,3}\s+[A-Z][a-z]+(\s+[A-Z][a-z-]+){2,}.*){2,}")
# Journal page headers glued into a sentence by PDF extraction.
_PAGE_HEADER = re.compile(r"\b(volume\s*\d+|issue\s*\d+|page:?\s*\d+|issn|e-issn|doi:|https?://|www\.)", re.IGNORECASE)
_CITATION_RUN = re.compile(r"(\[\d{1,3}\][^\[]{0,80}){2,}")
_TABLE_HEADER = re.compile(r"\b(gap category|description\s*&|implications\b.*\bdescription)", re.IGNORECASE)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")
_MIN_WORDS, _MAX_WORDS = 8, 70


@dataclass(frozen=True)
class GapStatementHit:
    sentence: str
    cue_class: str
    cue: str
    strength: float      # cue strength x section weight, in (0, 1]


def split_sentences(text: str) -> list[str]:
    text = " ".join(text.split())
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def section_weight(section_type: str, section_title: str = "") -> float:
    weight = SECTION_WEIGHTS.get((section_type or "UNKNOWN").upper(), 0.5)
    if _GAP_HEADING.search(section_title or ""):
        weight = max(weight, 0.9)
    return weight


def find_gap_statements(text: str, section_type: str, section_title: str = "") -> list[GapStatementHit]:
    """Gap statements in one chunk's text (strongest cue per sentence)."""
    weight = section_weight(section_type, section_title)
    if weight <= 0:
        return []
    hits: list[GapStatementHit] = []
    for sentence in split_sentences(text):
        words = len(sentence.split())
        if words < _MIN_WORDS or words > _MAX_WORDS:
            continue
        if _SELF_REFERENCE.search(sentence) or _CONTRIBUTION.search(sentence) or _MOTIVATION.search(sentence):
            continue
        if _PAGE_HEADER.search(sentence):
            continue  # extraction glued a journal header into the sentence
        if _CITATION_RUN.search(sentence) or _TABLE_HEADER.search(sentence) or sentence.lstrip().startswith("["):
            continue  # table rows / reference fragments
        capitalised = sum(1 for w in sentence.split() if w[:1].isupper())
        if capitalised / words > 0.55 or (words > 15 and capitalised / words > 0.42):
            continue  # title-case table or heading text, not prose
        if _TABLE_LIKE.search(sentence) or _TABLE_ROWS.search(sentence):
            continue  # comparison-table row flattened into one "sentence"
        best: Optional[GapStatementHit] = None
        for cue_class, strength, pattern in _CUES:
            m = pattern.search(sentence)
            if m and (best is None or strength > best.strength / weight):
                best = GapStatementHit(sentence, cue_class, m.group(0), round(strength * weight, 4))
        if best is not None:
            hits.append(best)
    return hits
