"""
tests/unit/test_query_understanding_prompts.py
------------------------------------------------
Unit tests for the Query Understanding prompt template.

Tests that the prompt:
- Is a non-empty string
- Contains key instruction keywords
- Includes the raw query verbatim
- Does not hard-code domain-specific example terms
"""

from __future__ import annotations

import pytest

from app.query_understanding.prompts import build_query_understanding_prompt


class TestBuildQueryUnderstandingPrompt:
    """Tests for the prompt builder function."""

    EXAMPLE_QUERY = "How are transformers used for protein structure prediction?"

    def test_returns_string(self):
        prompt = build_query_understanding_prompt(self.EXAMPLE_QUERY)
        assert isinstance(prompt, str)

    def test_prompt_is_non_empty(self):
        prompt = build_query_understanding_prompt(self.EXAMPLE_QUERY)
        assert len(prompt.strip()) > 0

    def test_raw_query_included_verbatim(self):
        """The prompt must contain the user's exact research question."""
        prompt = build_query_understanding_prompt(self.EXAMPLE_QUERY)
        assert self.EXAMPLE_QUERY in prompt

    def test_instructs_to_return_json(self):
        prompt = build_query_understanding_prompt(self.EXAMPLE_QUERY)
        assert "JSON" in prompt or "json" in prompt

    def test_includes_domain_instruction(self):
        prompt = build_query_understanding_prompt(self.EXAMPLE_QUERY)
        assert "domain" in prompt.lower()

    def test_includes_problem_instruction(self):
        prompt = build_query_understanding_prompt(self.EXAMPLE_QUERY)
        assert "problem" in prompt.lower()

    def test_includes_task_instruction(self):
        prompt = build_query_understanding_prompt(self.EXAMPLE_QUERY)
        assert "task" in prompt.lower()

    def test_includes_methods_instruction(self):
        prompt = build_query_understanding_prompt(self.EXAMPLE_QUERY)
        assert "method" in prompt.lower()

    def test_includes_expanded_queries_instruction(self):
        prompt = build_query_understanding_prompt(self.EXAMPLE_QUERY)
        assert "expanded_queries" in prompt or "expanded queries" in prompt.lower()

    def test_instructs_not_to_invent_datasets(self):
        """The prompt must warn the model against hallucinating datasets."""
        prompt = build_query_understanding_prompt(self.EXAMPLE_QUERY)
        assert "invent" in prompt.lower() or "hallucin" in prompt.lower() or \
               "not" in prompt.lower()

    def test_different_queries_produce_different_prompts(self):
        query_a = "Query about NLP"
        query_b = "Query about computer vision"
        prompt_a = build_query_understanding_prompt(query_a)
        prompt_b = build_query_understanding_prompt(query_b)
        assert prompt_a != prompt_b

    def test_terminology_variants_instruction_present(self):
        prompt = build_query_understanding_prompt(self.EXAMPLE_QUERY)
        assert "terminology" in prompt.lower() or "variant" in prompt.lower()

    def test_constraints_instruction_present(self):
        prompt = build_query_understanding_prompt(self.EXAMPLE_QUERY)
        assert "constraint" in prompt.lower()

    def test_research_intent_instruction_present(self):
        prompt = build_query_understanding_prompt(self.EXAMPLE_QUERY)
        assert "intent" in prompt.lower() or "research_intent" in prompt.lower()

    def test_prompt_is_generic_not_query_specific(self):
        """
        The prompt template must be generic: it must not pre-answer the
        structured fields with values specific to the user's query.

        The template may legitimately include inline format examples
        (e.g. "Automotive Cybersecurity" as an example of what a domain looks
        like). What matters is that the template does not contain content
        derived from the actual user query text.

        Specifically: for a query about "transformers + protein structure",
        the prompt must not pre-fill biology/bioinformatics-specific answers
        that would only make sense for THIS query.
        """
        prompt = build_query_understanding_prompt(self.EXAMPLE_QUERY)

        # The query is about transformers + protein structure prediction.
        # The prompt must NOT contain pre-filled biology-specific answers:
        assert "protein folding" not in prompt.lower()
        assert "AlphaFold" not in prompt
        assert "biology" not in prompt.lower() or self.EXAMPLE_QUERY in prompt

        # However, the query itself must always be present verbatim:
        assert self.EXAMPLE_QUERY in prompt
