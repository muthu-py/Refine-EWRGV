"""
app/gap_detection/prompts.py
------------------------------
Prompt templates for candidate gap detection and gap representation.

Design principles
-----------------
* Prompts are pure functions — given inputs they return a formatted string
  with no side effects.
* Every candidate gap must be grounded in passages that were actually
  provided, cited by their chunk IDs.  The detector later discards any
  cited ID that was not in the input.
* The model is told that "few papers" is not by itself a gap; the
  candidates are hypotheses that EWRGV validation will stress-test.
* JSON schema constraints are appended by the LLM provider layer, not here.
"""

from __future__ import annotations

from app.domain.enums import GapType, SectionType
from app.domain.models.paper_chunk import PaperChunk
from app.domain.models.query import ResearchQuery

_GAP_TYPE_DESCRIPTIONS: dict[GapType, str] = {
    GapType.METHODOLOGICAL: "no adequate method exists or methods have unaddressed weaknesses",
    GapType.POPULATION: "a specific population, language, or user group is understudied",
    GapType.DATASET: "suitable data or benchmarks are missing",
    GapType.GEOGRAPHIC: "geographic regions are underrepresented",
    GapType.TEMPORAL: "time periods or recent developments are not covered",
    GapType.CONTRADICTORY_EVIDENCE: "findings conflict without resolution",
    GapType.EVALUATION: "evaluation frameworks, metrics, or comparisons are lacking",
    GapType.OTHER: "does not fit the categories above",
}


def _format_gap_types() -> str:
    return "\n".join(
        f'- "{t.value}" — {_GAP_TYPE_DESCRIPTIONS[t]}' for t in GapType
    )


def _format_query_context(query: ResearchQuery) -> str:
    """Render the research question plus any Phase 2.1 understanding."""
    lines = [query.query]
    u = query.understanding
    if u is not None:
        context = [
            ("Domain", u.domain),
            ("Problem", u.problem),
            ("Task", u.task),
            ("Methods", ", ".join(u.methods)),
            ("Concepts", ", ".join(u.concepts)),
            ("Constraints", ", ".join(u.constraints)),
        ]
        rendered = [f"- {label}: {value}" for label, value in context if value]
        if rendered:
            lines.append("")
            lines.append("Structured understanding of the question:")
            lines.extend(rendered)
    return "\n".join(lines)


def _format_passages(chunks: list[PaperChunk], max_chunk_chars: int) -> str:
    blocks = []
    for chunk in chunks:
        text = chunk.text.strip()
        if len(text) > max_chunk_chars:
            text = text[:max_chunk_chars].rstrip() + " …"
        section = chunk.section_title or "Unknown section"
        if chunk.section_type != SectionType.UNKNOWN:
            section = f"{section} [{chunk.section_type.value}]"
        blocks.append(
            f"[chunk_id: {chunk.chunk_id}] (paper: {chunk.paper_id}; section: {section})\n{text}"
        )
    return "\n\n".join(blocks)


def build_gap_detection_prompt(
    query: ResearchQuery,
    chunks: list[PaperChunk],
    max_gaps: int,
    max_chunk_chars: int = 1500,
) -> str:
    """
    Build the prompt for the combined candidate gap detection call.

    One call returns, for each candidate gap: statement, gap type(s),
    supporting chunk IDs, initial confidence, and the structured
    representation fields (domain, population, methodology, ...).

    Parameters
    ----------
    query:
        The research query, optionally enriched by query understanding.
    chunks:
        Retrieved passages to analyse.  Their ``chunk_id`` values are the
        only IDs the model may cite.
    max_gaps:
        Upper bound on the number of candidates requested.
    max_chunk_chars:
        Passages longer than this are truncated in the prompt.
    """
    return f"""\
You are a research analyst identifying CANDIDATE research gaps from a set of \
retrieved literature passages. Your candidates will later be independently \
validated against a wider literature search, so they are hypotheses, not conclusions.

## Research Question

{_format_query_context(query)}

## Retrieved Passages

{_format_passages(chunks, max_chunk_chars)}

## Instructions

Identify at most {max_gaps} candidate research gaps relevant to the research question \
that are grounded in the passages above. Return them in the "candidate_gaps" list. \
Return an empty list if the passages do not indicate any gap.

Good signals of a gap include: stated limitations, stated future work, open problems, \
untested settings or populations, missing datasets or evaluations, and conflicting \
findings between passages. Passages from "Limitations", "Future Work", "Discussion", \
and "Conclusion" sections are especially informative.

For each candidate gap provide:

### gap_statement
One specific, self-contained sentence describing what is missing or unresolved. \
Avoid generic statements such as "more research is needed".

### gap_type
The single best-fitting type, exactly one of:
{_format_gap_types()}

### secondary_gap_types
Other types from the same list that also apply (may be empty). Do not repeat gap_type.

### supporting_chunk_ids
The chunk_id values of the passages above that motivate this gap. Copy them exactly. \
Cite at least one. NEVER cite an ID that does not appear above.

### contradicting_chunk_ids
The chunk_id values of passages above that suggest this gap may ALREADY be addressed \
(e.g. a passage reporting a method, dataset, or evaluation that fills it). Return an \
empty list if none do. Do not list a passage here and in supporting_chunk_ids. \
NEVER cite an ID that does not appear above.

### initial_confidence
A number from 0.0 to 1.0: how strongly the cited passages themselves indicate this gap \
(e.g. an explicit limitation statement → high; an inference from absence → low). \
This is NOT a judgement of novelty or of whether the gap exists in the wider literature.

### rationale
One short sentence explaining how the cited passages indicate the gap.

### domain, population, methodology, outcome, temporal_scope, geographic_scope
Decompose the gap into these components. Use null when a component does not apply.

### key_concepts
The core concepts a literature search would need to verify this gap.

### synonyms
Alternative terms, abbreviations, and phrasings for the key concepts.

## Important constraints

- A small number of passages on a topic is NOT by itself a research gap.
- Do NOT invent papers, results, datasets, or claims that are not in the passages.
- Do NOT duplicate gaps; merge candidates that describe the same gap.
- Return ONLY valid JSON. No markdown, no explanation outside the JSON object.
"""


def build_gap_representation_prompt(gap_statement: str) -> str:
    """
    Build the prompt for decomposing a single gap statement into a
    StructuredGapRepresentation (used by GapRepresentationBuilder.build()).
    """
    return f"""\
You are a research analyst decomposing a research gap statement into structured \
components that will be used to generate literature verification queries.

## Gap Statement

{gap_statement}

## Instructions

Return a JSON object with these fields:

- domain: the research domain, or null.
- population: the population, language, or user group concerned, or null.
- methodology: the method or technique concerned, or null.
- outcome: the outcome, task, or property concerned, or null.
- temporal_scope: any time period the gap is restricted to, or null.
- geographic_scope: any geographic region the gap is restricted to, or null.
- key_concepts: the core concepts a literature search would need to verify this gap.
- synonyms: alternative terms, abbreviations, and phrasings for the key concepts.

## Important constraints

- Only use information stated in or directly implied by the gap statement.
- Use null (strings) or an empty list (lists) when a field does not apply.
- Return ONLY valid JSON. No markdown, no explanation outside the JSON object.
"""
