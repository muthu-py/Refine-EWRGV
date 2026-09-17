"""
app/retrieval/knowledge_graph.py
----------------------------------
Knowledge Graph retrieval — concept-based document chunk retrieval.

Depends on:
    KnowledgeGraphStore  (app.domain.interfaces) — for entity/concept lookup
    CorpusStore          (app.domain.interfaces) — for chunk resolution

Flow:
    query → extract matching entities/concepts from KG → traverse
    relationships to find related entities → collect chunk IDs from
    matched + related entities → resolve chunks → rank by connection
    strength → return top-K results.

Scoring heuristic:
    Each chunk receives a score based on how it was discovered:
    - Direct entity match:   1.0 per matching entity
    - Related entity (1-hop): 0.5 per relationship traversal
    Scores are summed if a chunk appears via multiple entities.
    Final scores are normalised to [0, 1].
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.domain.interfaces import CorpusStore, KnowledgeGraphStore
from app.domain.models.paper import DocumentChunk
from app.domain.models.retrieval_result import RetrievalResult
from app.retrieval.interfaces import ScoredChunk

logger = get_logger(__name__)


class KnowledgeGraphRetriever:
    """
    Retrieves document chunks by querying a knowledge graph of concepts.

    Implementation
    --------------
    1. Find entities/concepts matching the query via the KG store.
    2. For each matched entity, collect its directly linked chunk IDs.
    3. Traverse 1-hop relationships to find related entities.
    4. Collect chunk IDs from related entities (lower weight).
    5. Aggregate scores per chunk (direct match = 1.0, related = 0.5).
    6. Resolve chunk IDs to full DocumentChunks via the corpus store.
    7. Return top-K ranked results.

    Parameters
    ----------
    kg_store:
        Any object satisfying the KnowledgeGraphStore protocol.
    corpus_store:
        Any object satisfying the CorpusStore protocol.
    """

    def __init__(
        self,
        kg_store: KnowledgeGraphStore,
        corpus_store: CorpusStore,
    ) -> None:
        self._kg = kg_store
        self._corpus = corpus_store
        # Build a chunk lookup for fast resolution
        self._chunk_map: dict[str, DocumentChunk] | None = None

    def _ensure_chunk_map(self) -> dict[str, DocumentChunk]:
        """Build or return cached chunk lookup."""
        if self._chunk_map is None:
            self._chunk_map = {
                c.chunk_id: c for c in self._corpus.get_all_chunks()
            }
        return self._chunk_map

    def retrieve(self, query: str, top_k: int = 20) -> list[DocumentChunk]:
        """
        Satisfy the Retriever interface.

        Returns plain DocumentChunk objects without scores.
        """
        results = self.retrieve_results(query, top_k)
        return [
            DocumentChunk(
                chunk_id=r.chunk_id,
                paper_id=r.paper_id,
                section=r.section,
                text=r.text,
            )
            for r in results
        ]

    def retrieve_scored(self, query: str, top_k: int = 20) -> list[ScoredChunk]:
        """
        Return KG-scored chunks for internal fusion use.

        Satisfies the ScoredRetriever protocol (app.retrieval.interfaces).
        """
        chunk_map = self._ensure_chunk_map()

        # Step 1: Find matching entities
        matched_entities = self._kg.find_entities(query)
        if not matched_entities:
            logger.debug("KnowledgeGraphRetriever: no entities matched query")
            return []

        # Step 2+3: Collect chunk scores with provenance tracking
        # chunk_id → (accumulated_score, matched_entity_names, relations)
        chunk_scores: dict[str, tuple[float, list[str], list[str]]] = {}

        for entity in matched_entities:
            entity_id = entity["entity_id"]
            entity_name = entity["name"]

            # Direct chunks from this entity (score = 1.0 per entity)
            direct_chunks = self._kg.get_entity_chunks(entity_id)
            for cid in direct_chunks:
                if cid in chunk_scores:
                    score, entities, relations = chunk_scores[cid]
                    chunk_scores[cid] = (
                        score + 1.0,
                        entities + [entity_name],
                        relations,
                    )
                else:
                    chunk_scores[cid] = (1.0, [entity_name], [])

            # Related entities (1-hop traversal, score = 0.5 per relation)
            related = self._kg.get_related_entities(entity_id)
            for rel in related:
                target_id = rel["target_id"]
                relation = rel["relation"]
                target_name = rel.get("target_name", target_id)

                related_chunks = self._kg.get_entity_chunks(target_id)
                for cid in related_chunks:
                    relation_label = f"{entity_name} --{relation}--> {target_name}"
                    if cid in chunk_scores:
                        score, entities, relations = chunk_scores[cid]
                        chunk_scores[cid] = (
                            score + 0.5,
                            entities,
                            relations + [relation_label],
                        )
                    else:
                        chunk_scores[cid] = (0.5, [], [relation_label])

        if not chunk_scores:
            return []

        # Step 4: Normalise scores to [0, 1]
        max_score = max(s for s, _, _ in chunk_scores.values())
        if max_score <= 0:
            return []

        # Step 5: Build sorted result list
        scored_items: list[tuple[str, float, list[str], list[str]]] = []
        for cid, (raw_score, entities, relations) in chunk_scores.items():
            normalised = raw_score / max_score
            scored_items.append((cid, normalised, entities, relations))

        # Sort by score descending, then by chunk_id for determinism
        scored_items.sort(key=lambda x: (-x[1], x[0]))

        # Top-K
        top = scored_items[:top_k]

        # Step 6: Resolve chunks and build ScoredChunk list
        results: list[ScoredChunk] = []
        for rank, (cid, score, entities, relations) in enumerate(top):
            chunk = chunk_map.get(cid)
            if chunk is None:
                continue
            results.append(
                ScoredChunk(
                    chunk=chunk,
                    score=score,
                    retriever="knowledge_graph",
                    rank=rank + 1,
                )
            )

        return results

    def retrieve_results(self, query: str, top_k: int = 20) -> list[RetrievalResult]:
        """
        Retrieve the top-K knowledge-graph-ranked chunks as RetrievalResult objects.

        Parameters
        ----------
        query:
            The search query string.
        top_k:
            Number of top results to return.

        Returns
        -------
        list[RetrievalResult]
            Ranked list of retrieval results with
            ``retrieval_method="knowledge_graph"``.
            Metadata includes ``matched_entities`` and ``relations``.
        """
        chunk_map = self._ensure_chunk_map()

        # Reuse the scoring logic but also capture metadata
        matched_entities = self._kg.find_entities(query)
        if not matched_entities:
            return []

        # chunk_id → (score, matched_entities, relations)
        chunk_scores: dict[str, tuple[float, list[str], list[str]]] = {}

        for entity in matched_entities:
            entity_id = entity["entity_id"]
            entity_name = entity["name"]

            direct_chunks = self._kg.get_entity_chunks(entity_id)
            for cid in direct_chunks:
                if cid in chunk_scores:
                    s, e, r = chunk_scores[cid]
                    if entity_name not in e:
                        chunk_scores[cid] = (s + 1.0, e + [entity_name], r)
                    else:
                        chunk_scores[cid] = (s + 1.0, e, r)
                else:
                    chunk_scores[cid] = (1.0, [entity_name], [])

            related = self._kg.get_related_entities(entity_id)
            for rel in related:
                target_id = rel["target_id"]
                relation = rel["relation"]
                target_name = rel.get("target_name", target_id)
                relation_label = f"{entity_name} --{relation}--> {target_name}"

                related_chunks = self._kg.get_entity_chunks(target_id)
                for cid in related_chunks:
                    if cid in chunk_scores:
                        s, e, r = chunk_scores[cid]
                        if relation_label not in r:
                            chunk_scores[cid] = (s + 0.5, e, r + [relation_label])
                        else:
                            chunk_scores[cid] = (s + 0.5, e, r)
                    else:
                        chunk_scores[cid] = (0.5, [], [relation_label])

        if not chunk_scores:
            return []

        max_score = max(s for s, _, _ in chunk_scores.values())
        if max_score <= 0:
            return []

        scored_items = [
            (cid, raw / max_score, ents, rels)
            for cid, (raw, ents, rels) in chunk_scores.items()
        ]
        scored_items.sort(key=lambda x: (-x[1], x[0]))
        top = scored_items[:top_k]

        results: list[RetrievalResult] = []
        for rank, (cid, score, entities, relations) in enumerate(top):
            chunk = chunk_map.get(cid)
            if chunk is None:
                continue
            results.append(
                RetrievalResult(
                    chunk_id=cid,
                    paper_id=chunk.paper_id,
                    section=chunk.section,
                    text=chunk.text,
                    score=score,
                    rank=rank + 1,
                    retrieval_method="knowledge_graph",
                    metadata={
                        "matched_entities": entities,
                        "relations": relations,
                    },
                )
            )

        return results
