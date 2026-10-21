"""Small, dependency-free evaluators using TREC-style relevance semantics."""

from __future__ import annotations

import math
from collections.abc import Sequence


def evaluate_ranking(ranked_ids: Sequence[str], qrels: dict[str, int], *, cutoff: int = 10) -> dict[str, float | int]:
    """Return P@5, P@10, R@10, reciprocal rank, AP and nDCG@10.

    Binary relevance for P/R/RR/AP is grade > 0. nDCG uses the TREC
    graded-relevance gain 2**grade - 1. Unjudged retrieved documents count as
    nonrelevant for binary measures, matching trec_eval's default convention;
    judged and unjudged hits are reported so that assumption is visible.
    """
    ids = [str(doc_id) for doc_id in ranked_ids]
    relevant = {str(doc_id) for doc_id, grade in qrels.items() if int(grade) > 0}
    judged = {str(doc_id) for doc_id in qrels}
    seen: set[str] = set()
    deduped: list[str] = []
    for doc_id in ids:
        if doc_id not in seen:
            deduped.append(doc_id)
            seen.add(doc_id)

    def precision(k: int) -> float:
        return sum(doc_id in relevant for doc_id in deduped[:k]) / k

    hits = [doc_id for doc_id in deduped if doc_id in relevant]
    rr = 1.0 / (deduped.index(hits[0]) + 1) if hits else 0.0
    precisions = [sum(doc_id in relevant for doc_id in deduped[:rank]) / rank
                  for rank, doc_id in enumerate(deduped, 1) if doc_id in relevant]
    average_precision = sum(precisions) / len(relevant) if relevant else 0.0

    def dcg(grades: Sequence[int]) -> float:
        return sum((2 ** int(grade) - 1) / math.log2(rank + 1)
                   for rank, grade in enumerate(grades, 1))

    actual_grades = [int(qrels.get(doc_id, 0)) for doc_id in deduped[:cutoff]]
    ideal_grades = sorted((int(g) for g in qrels.values()), reverse=True)[:cutoff]
    ideal_dcg = dcg(ideal_grades)
    ndcg = dcg(actual_grades) / ideal_dcg if ideal_dcg else 0.0

    return {
        "P@5": precision(5),
        "P@10": precision(10),
        "R@10": sum(doc_id in relevant for doc_id in deduped[:10]) / len(relevant) if relevant else 0.0,
        "RR": rr,
        "AP": average_precision,
        f"nDCG@{cutoff}": ndcg,
        "judged_retrieved@10": sum(doc_id in judged for doc_id in deduped[:10]),
        "unjudged_retrieved@10": sum(doc_id not in judged for doc_id in deduped[:10]),
        "relevant_judgments": len(relevant),
    }


def aggregate(per_query: list[dict[str, float | int]]) -> dict[str, float | int]:
    """Macro-average ranking metrics and sum the judgment coverage counts."""
    if not per_query:
        return {"query_count": 0}
    metric_names = ["P@5", "P@10", "R@10", "RR", "AP", "nDCG@10"]
    result: dict[str, float | int] = {name: sum(float(row[name]) for row in per_query) / len(per_query)
                                      for name in metric_names}
    result.update({
        "query_count": len(per_query),
        "judged_retrieved@10": sum(int(row["judged_retrieved@10"]) for row in per_query),
        "unjudged_retrieved@10": sum(int(row["unjudged_retrieved@10"]) for row in per_query),
        "relevant_judgments": sum(int(row["relevant_judgments"]) for row in per_query),
    })
    return result
