"""
app/domain/models/retrieval_result.py
--------------------------------------
Unified domain model for retrieval results.

All retrieval methods (dense, sparse, knowledge graph, hybrid/fusion)
return instances of RetrievalResult.  This keeps downstream consumers
(gap detection, validation, API) decoupled from retrieval internals.

The model is intentionally independent of any specific retrieval
implementation or scoring mechanism.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    """
    A single retrieval result representing a document chunk matched to a query.

    Attributes
    ----------
    chunk_id:
        Unique identifier for the retrieved chunk.
    paper_id:
        Identifier of the parent paper.
    section:
        The section heading this chunk was extracted from.
    text:
        The actual text content of the chunk.
    score:
        Retrieval score (method-specific; higher is better).
    rank:
        1-based rank within the result list.
    retrieval_method:
        Which retrieval method produced this result:
        ``"dense"``, ``"sparse"``, ``"knowledge_graph"``, or ``"hybrid"``.
    metadata:
        Optional extra information.  May include:
        - ``matched_entities``: list of KG entities matched (KG retrieval)
        - ``relation``: KG relationship label (KG retrieval)
        - ``contributing_methods``: list of methods that contributed (hybrid)
        - ``source``: provenance detail
        - ``explanation``: human-readable scoring rationale
    """

    chunk_id: str
    paper_id: str
    section: str = ""
    text: str = ""
    score: float = 0.0
    rank: int = 0
    retrieval_method: str = ""
    metadata: dict = Field(default_factory=dict)
