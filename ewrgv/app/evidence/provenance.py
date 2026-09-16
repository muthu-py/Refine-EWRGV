"""
app/evidence/provenance.py
---------------------------
Evidence provenance tracking.

Records which search type, query, and retrieval rank produced each
evidence item.  Used for audit trails and explanation generation.

Status: SKELETON – placeholder module.
"""

from __future__ import annotations

from app.domain.models.evidence import Evidence, Provenance


def attach_provenance(
    evidence: Evidence,
    search_type: str,
    query_used: str,
    retrieval_rank: int | None = None,
    rerank_score: float | None = None,
) -> Evidence:
    """
    Attach provenance metadata to an Evidence item.

    Returns the same Evidence object with its provenance field set.
    """
    evidence.provenance = Provenance(
        search_type=search_type,
        query_used=query_used,
        retrieval_rank=retrieval_rank,
        rerank_score=rerank_score,
    )
    return evidence
