"""
app/query_understanding/prompts.py
------------------------------------
Prompt template for the Query Understanding & Expansion LLM call.

Design principles
-----------------
* The prompt is a pure function — given a raw query string it returns a
  formatted prompt string with no side effects.
* The prompt is separated from the service so it can be tested, versioned,
  and swapped independently.
* Instructions are explicit and conservative: the model is told NOT to invent
  datasets, papers, or results merely because they are common in the domain.
* JSON schema constraints are appended by the LLM provider layer, not here.
"""

from __future__ import annotations


def build_query_understanding_prompt(raw_query: str) -> str:
    """
    Build the full prompt for the Query Understanding & Expansion task.

    The prompt instructs the LLM to decompose a research question into
    structured fields and generate multiple complementary search queries
    for high-recall literature discovery.

    Parameters
    ----------
    raw_query:
        The verbatim research question provided by the user.

    Returns
    -------
    str
        A ready-to-send prompt string.
    """
    return f"""\
You are a research assistant helping to understand and expand a user's research question \
for systematic literature discovery.

## Research Question

{raw_query}

## Instructions

Analyse the research question above and produce a structured JSON response with the \
following fields. Follow each instruction precisely.

### domain
Identify the high-level research domain (e.g., "Automotive Cybersecurity", \
"Natural Language Processing", "Computer Vision"). Return a single string or null.

### problem
Identify the central problem or challenge the question is about. Return a concise \
string or null.

### task
Identify the specific machine learning or research task (e.g., "intrusion detection", \
"named entity recognition", "image segmentation"). Return a single string or null.

### methods
List the methods, techniques, or algorithms explicitly mentioned or strongly implied \
by the question. Return a list of strings, or an empty list if none are clear.
Do NOT include methods that are merely popular in the domain but unrelated to the question.

### concepts
List the important conceptual terms that a literature search should include. \
Include both general and specific concepts that are central to the question.

### entities
List specific named entities such as organisations, standards, products, protocols, \
or proper names (e.g., "CAN bus", "BERT", "ImageNet"). Return an empty list if none \
are clearly present.

### datasets
List benchmark or evaluation datasets ONLY if explicitly named in the question or \
unambiguously implied by it. Return an empty list otherwise.
Do NOT invent dataset names based on domain conventions.

### metrics
List evaluation metrics ONLY if explicitly mentioned or unambiguously implied by the \
research objective (e.g., "F1-score", "accuracy", "detection rate"). Return an empty \
list otherwise.

### constraints
List any explicit scope constraints mentioned (e.g., geographic region, population \
subgroup, timeframe, data modality, technology stack, application domain). \
Return an empty list if none are stated.

### research_intent
Characterise the high-level intent of the question. Choose the most appropriate of:
- "method exploration" — asking what methods exist or are used
- "survey" — seeking a broad overview of a field
- "comparative study" — comparing approaches or results
- "dataset or benchmark" — focused on evaluation resources
- "application" — applying existing techniques to a new context
- "gap identification" — looking for what is missing in the literature
- "other" — does not fit the above
Return a single string.

### terminology_variants
Generate a list of alternative terms, abbreviations, and synonyms for the key \
concepts that would expand literature search coverage. Think about how different \
authors might refer to the same idea.

### method_variants
Generate a list of alternative names for the methods identified above (e.g., \
alternative abbreviations, predecessor/successor techniques, related model families). \
Only include variants that are genuinely related to this question. Return an empty \
list if no method variants are applicable.

### expanded_queries
Generate {_MAX_QUERIES_PLACEHOLDER} diverse search query strings suitable for \
querying academic literature databases (e.g., Semantic Scholar, arXiv). Each query \
should be a short keyword-style string (not a full sentence). Aim for diversity: \
cover different terminology, method names, and domain perspectives. These queries \
will be used for high-recall literature discovery — prioritise coverage over precision.

## Important constraints

- Do NOT invent specific papers, authors, results, or factual claims.
- Do NOT assume datasets exist unless clearly stated.
- If a field cannot be reasonably inferred from the question, return null (for \
  string fields) or an empty list (for list fields).
- Return ONLY valid JSON. No markdown, no explanation outside the JSON object.
"""


# Internal constant referenced by the prompt builder.
_MAX_QUERIES_PLACEHOLDER = "between 5 and 8"
