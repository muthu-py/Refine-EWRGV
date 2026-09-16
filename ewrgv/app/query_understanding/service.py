"""
app/query_understanding/service.py
------------------------------------
QueryUnderstandingService — Phase 2.1 core business logic.

Responsibilities
----------------
1. Accept a ResearchQuery with a raw question.
2. Build the LLM prompt via the prompt template.
3. Call the injected LLMProvider.complete_structured().
4. Validate the raw LLM response via QueryUnderstandingLLMOutput.
5. Map the validated response to the QueryUnderstanding domain model.
6. Return the enriched ResearchQuery (understanding + expanded_queries set).

Architecture position
---------------------
    ResearchPipeline / API route
              ↓
    QueryUnderstandingService      ← this file
              ↓
    LLMProvider (protocol)
              ↓
    configured LLM implementation

The service depends on the LLMProvider *interface* only, never on a
concrete SDK or vendor class.

Error handling
--------------
* pydantic.ValidationError from schema validation → ProviderError
* ProviderError from the LLM provider → re-raised as-is
* Any other unexpected exception → wrapped in ProviderError
"""

from __future__ import annotations

import json

import pydantic

from app.core.exceptions import ProviderError
from app.core.logging import get_logger
from app.domain.interfaces import LLMProvider
from app.domain.models.query import ResearchQuery
from app.domain.models.query_understanding import QueryUnderstanding
from app.query_understanding.prompts import build_query_understanding_prompt
from app.query_understanding.schemas import QueryUnderstandingLLMOutput

logger = get_logger(__name__)

# Maximum number of expanded queries we accept from the LLM output.
# The prompt asks for 5–8; this guard prevents pathological edge cases.
_MAX_EXPANDED_QUERIES = 15


class QueryUnderstandingService:
    """
    Orchestrates the query understanding and expansion step.

    Parameters
    ----------
    llm:
        Any object satisfying the LLMProvider protocol.
        Typically an OpenAILLMProvider or a mock in tests.
    max_expanded_queries:
        Hard upper bound on accepted expanded queries.  Defaults to the
        module-level constant.
    """

    def __init__(
        self,
        llm: LLMProvider,
        max_expanded_queries: int = _MAX_EXPANDED_QUERIES,
    ) -> None:
        self._llm = llm
        self._max_expanded_queries = max_expanded_queries

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self, research_query: ResearchQuery) -> ResearchQuery:
        """
        Run query understanding and expansion on a ResearchQuery.

        The input ResearchQuery is treated as immutable — a new instance is
        returned with ``understanding`` and ``expanded_queries`` populated.
        The original ``query`` string is always preserved unchanged.

        Parameters
        ----------
        research_query:
            The research query to enrich.  Must have a non-empty ``query``.

        Returns
        -------
        ResearchQuery
            A new ResearchQuery instance with understanding and
            expanded_queries populated.

        Raises
        ------
        ProviderError
            On any LLM provider failure or invalid/unparseable LLM output.
        """
        raw_query = research_query.query
        logger.info(
            "QueryUnderstandingService: starting",
            extra={
                "query_id": research_query.query_id,
                "provider": type(self._llm).__name__,
            },
        )

        # Step 1 – build the prompt
        prompt = build_query_understanding_prompt(raw_query)

        # Step 2 – call the LLM
        llm_output_dict = self._call_llm(prompt, research_query.query_id)

        # Step 3 – validate the raw dict
        llm_output = self._validate_llm_output(
            llm_output_dict, research_query.query_id
        )

        # Step 4 – map to domain objects
        understanding = self._map_to_domain(llm_output)
        expanded_queries = self._cap_expanded_queries(llm_output.expanded_queries)

        logger.info(
            "QueryUnderstandingService: completed successfully",
            extra={
                "query_id": research_query.query_id,
                "expanded_queries_count": len(expanded_queries),
                "domain": understanding.domain,
            },
        )

        # Step 5 – return enriched ResearchQuery (immutable model_copy)
        return research_query.model_copy(
            update={
                "understanding": understanding,
                "expanded_queries": expanded_queries,
            }
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _call_llm(self, prompt: str, query_id: str) -> dict:
        """Call the LLM provider and return the raw response dict."""
        schema = QueryUnderstandingLLMOutput.json_schema_for_llm()
        try:
            result = self._llm.complete_structured(prompt, schema)
            if not isinstance(result, dict):
                raise ProviderError(
                    "LLM provider returned a non-dict response from "
                    "complete_structured().",
                    detail=f"Type received: {type(result).__name__}",
                )
            return result
        except ProviderError:
            logger.warning(
                "QueryUnderstandingService: provider failure",
                extra={"query_id": query_id},
            )
            raise
        except Exception as exc:
            logger.warning(
                "QueryUnderstandingService: unexpected error from LLM",
                extra={"query_id": query_id},
            )
            raise ProviderError(
                f"Unexpected error during LLM call: {exc}",
                detail=str(exc),
            ) from exc

    def _validate_llm_output(
        self, raw: dict, query_id: str
    ) -> QueryUnderstandingLLMOutput:
        """Validate the raw dict against the LLM output schema."""
        try:
            return QueryUnderstandingLLMOutput.model_validate(raw)
        except pydantic.ValidationError as exc:
            logger.warning(
                "QueryUnderstandingService: LLM output failed validation",
                extra={"query_id": query_id, "errors": exc.error_count()},
            )
            raise ProviderError(
                "LLM returned invalid structured output for query understanding.",
                detail=str(exc),
            ) from exc

    def _map_to_domain(
        self, llm_output: QueryUnderstandingLLMOutput
    ) -> QueryUnderstanding:
        """Map a validated LLM output to the domain QueryUnderstanding model."""
        return QueryUnderstanding(
            domain=llm_output.domain,
            problem=llm_output.problem,
            task=llm_output.task,
            methods=llm_output.methods,
            concepts=llm_output.concepts,
            entities=llm_output.entities,
            datasets=llm_output.datasets,
            metrics=llm_output.metrics,
            constraints=llm_output.constraints,
            research_intent=llm_output.research_intent,
            terminology_variants=llm_output.terminology_variants,
            method_variants=llm_output.method_variants,
        )

    def _cap_expanded_queries(self, queries: list[str]) -> list[str]:
        """
        Apply the hard cap on expanded queries and filter empty strings.

        Queries are already cleaned by the schema validator, but we apply
        the count cap here after full validation.
        """
        valid = [q for q in queries if q.strip()]
        return valid[: self._max_expanded_queries]
