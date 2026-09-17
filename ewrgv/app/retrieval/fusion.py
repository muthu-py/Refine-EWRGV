"""
app/retrieval/fusion.py
------------------------
Reciprocal Rank Fusion (RRF) for combining multiple retrieval results.

Combines results from:
    - Dense (semantic) retrieval
    - Sparse (BM25) retrieval
    - Knowledge Graph retrieval

RRF Formula
-----------
    score(d) = Σ  1 / (k + rank_i(d))

    where:
    - k is the RRF smoothing constant (default 60, configurable)
    - rank_i(d) is the 1-based rank of document d in ranked list i
    - the sum is over all ranked lists that contain document d

RRF is used because the three retrieval methods produce scores on
fundamentally different scales (cosine similarity vs BM25 vs KG
connection strength).  RRF normalises this by using rank positions only.

Reference:
    Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009).
    "Reciprocal Rank Fusion outperforms Condorcet and individual
    Rank Learning Methods." SIGIR '09.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.models.retrieval_result import RetrievalResult
from app.retrieval.interfaces import ScoredChunk

logger = get_logger(__name__)

# Default RRF smoothing constant.  Higher values reduce the influence
# of individual high ranks; 60 is the standard value from the literature.
DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[ScoredChunk]],
    top_k: int = 20,
    k: int = DEFAULT_RRF_K,
) -> list[ScoredChunk]:
    """
    Merge multiple ranked lists into one using Reciprocal Rank Fusion.

    Parameters
    ----------
    ranked_lists:
        Each inner list is a ranked result from one retriever.
        Items must have a ``rank`` attribute (1-based).
    top_k:
        Number of results to return.
    k:
        RRF smoothing constant (default 60).

    Returns
    -------
    list[ScoredChunk]:
        Fused and re-ranked list of ScoredChunks with RRF scores.
    """
    # chunk_id → accumulated RRF score
    rrf_scores: dict[str, float] = {}
    # chunk_id → best ScoredChunk (for metadata)
    best_chunk: dict[str, ScoredChunk] = {}
    # chunk_id → set of contributing retrievers
    contributors: dict[str, list[str]] = {}

    for ranked_list in ranked_lists:
        for sc in ranked_list:
            cid = sc.chunk.chunk_id
            rrf_contribution = 1.0 / (k + sc.rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_contribution

            # Keep the chunk reference (use first seen)
            if cid not in best_chunk:
                best_chunk[cid] = sc

            # Track contributing methods
            if cid not in contributors:
                contributors[cid] = []
            if sc.retriever and sc.retriever not in contributors[cid]:
                contributors[cid].append(sc.retriever)

    # Sort by RRF score descending, then chunk_id for determinism
    sorted_items = sorted(
        rrf_scores.items(),
        key=lambda x: (-x[1], x[0]),
    )

    # Top-K
    top = sorted_items[:top_k]

    return [
        ScoredChunk(
            chunk=best_chunk[cid].chunk,
            score=score,
            retriever="hybrid",
            rank=rank + 1,
        )
        for rank, (cid, score) in enumerate(top)
    ]


def reciprocal_rank_fusion_results(
    result_lists: list[list[RetrievalResult]],
    top_k: int = 20,
    k: int = DEFAULT_RRF_K,
) -> list[RetrievalResult]:
    """
    Merge multiple RetrievalResult lists using Reciprocal Rank Fusion.

    This is the primary fusion function for combining dense, sparse, and
    KG retrieval results into a single unified result list.

    Parameters
    ----------
    result_lists:
        Each inner list is a ranked result from one retrieval method.
        Items must have ``rank`` (1-based) and ``chunk_id`` attributes.
    top_k:
        Number of results to return.
    k:
        RRF smoothing constant (default 60).

    Returns
    -------
    list[RetrievalResult]:
        Fused results with ``retrieval_method="hybrid"``.
        Metadata includes ``contributing_methods`` and ``rrf_k``.
    """
    # chunk_id → accumulated RRF score
    rrf_scores: dict[str, float] = {}
    # chunk_id → best result (for text/metadata)
    best_result: dict[str, RetrievalResult] = {}
    # chunk_id → contributing methods with their ranks
    contributors: dict[str, list[dict]] = {}

    for result_list in result_lists:
        for result in result_list:
            cid = result.chunk_id
            rrf_contribution = 1.0 / (k + result.rank)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_contribution

            if cid not in best_result:
                best_result[cid] = result

            if cid not in contributors:
                contributors[cid] = []
            contributors[cid].append({
                "method": result.retrieval_method,
                "rank": result.rank,
                "score": result.score,
            })

    # Sort by RRF score descending, then chunk_id for determinism
    sorted_items = sorted(
        rrf_scores.items(),
        key=lambda x: (-x[1], x[0]),
    )

    # Top-K
    top = sorted_items[:top_k]

    results: list[RetrievalResult] = []
    for rank, (cid, rrf_score) in enumerate(top):
        source = best_result[cid]
        # Merge metadata from source with fusion provenance
        merged_metadata = dict(source.metadata) if source.metadata else {}
        merged_metadata["contributing_methods"] = contributors.get(cid, [])
        merged_metadata["rrf_k"] = k

        results.append(
            RetrievalResult(
                chunk_id=cid,
                paper_id=source.paper_id,
                section=source.section,
                text=source.text,
                score=rrf_score,
                rank=rank + 1,
                retrieval_method="hybrid",
                metadata=merged_metadata,
            )
        )

    return results


class HybridRetriever:
    """
    Combines Dense, Sparse, and Knowledge Graph retrieval via RRF.

    Composes three retrievers and fuses their results using Reciprocal
    Rank Fusion.  Each retriever is called independently with its own
    top_k, and the fused results are trimmed to the final top_k.

    Parameters
    ----------
    dense_retriever:
        Must have a ``retrieve_results(query, top_k)`` method.
    sparse_retriever:
        Must have a ``retrieve_results(query, top_k)`` method.
    kg_retriever:
        Must have a ``retrieve_results(query, top_k)`` method.
        May be None if KG retrieval is not available.
    top_k:
        Default number of final results to return.
    rrf_k:
        RRF smoothing constant (default 60).
    per_retriever_k:
        Number of results to request from each individual retriever.
        Defaults to 2× the final top_k to ensure good fusion coverage.
    """

    def __init__(
        self,
        dense_retriever: object,
        sparse_retriever: object,
        kg_retriever: object | None = None,
        top_k: int = 20,
        rrf_k: int = DEFAULT_RRF_K,
        per_retriever_k: int | None = None,
    ) -> None:
        self._dense = dense_retriever
        self._sparse = sparse_retriever
        self._kg = kg_retriever
        self._top_k = top_k
        self._rrf_k = rrf_k
        self._per_k = per_retriever_k or (top_k * 2)

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """
        Run all retrieval methods and fuse results via RRF.

        Parameters
        ----------
        query:
            The search query string.
        top_k:
            Number of final results.  Defaults to constructor value.

        Returns
        -------
        list[RetrievalResult]
            Fused results with ``retrieval_method="hybrid"``.
        """
        final_k = top_k or self._top_k

        # Collect results from each retriever
        result_lists: list[list[RetrievalResult]] = []

        # Dense retrieval
        dense_results = self._dense.retrieve_results(query, self._per_k)
        if dense_results:
            result_lists.append(dense_results)
            logger.debug("HybridRetriever: dense returned %d results", len(dense_results))

        # Sparse retrieval
        sparse_results = self._sparse.retrieve_results(query, self._per_k)
        if sparse_results:
            result_lists.append(sparse_results)
            logger.debug("HybridRetriever: sparse returned %d results", len(sparse_results))

        # KG retrieval (optional)
        if self._kg is not None:
            kg_results = self._kg.retrieve_results(query, self._per_k)
            if kg_results:
                result_lists.append(kg_results)
                logger.debug("HybridRetriever: KG returned %d results", len(kg_results))

        if not result_lists:
            logger.warning("HybridRetriever: no results from any retriever")
            return []

        # Fuse via RRF
        fused = reciprocal_rank_fusion_results(
            result_lists,
            top_k=final_k,
            k=self._rrf_k,
        )

        logger.info(
            "HybridRetriever: fused %d results from %d methods",
            len(fused),
            len(result_lists),
        )
        return fused
