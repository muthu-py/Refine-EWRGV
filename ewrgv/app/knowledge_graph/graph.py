"""
app/knowledge_graph/graph.py
------------------------------
The run's Academic Knowledge Graph: data model, NetworkX view, exports.

Stored as one JSON payload per run (table kg_graphs), so it can be reloaded
without re-tagging.  Node kinds and edges:

    Paper     -MENTIONS(papers' chunk count)->  Entity
    Paper     -STATES->                          GapStatement
    GapStatement -ABOUT->                        Entity
    Paper     -REPORTS->                         ResultClaim -ON-> Entity
    Entity    -CO_OCCURS(papers, PMI)-           Entity      (undirected, paper level)
    Entity    -IN_COMMUNITY->                    Community   (Louvain over CO_OCCURS)

Chunk-level mentions are kept in ``mentions`` (chunk_id -> entity counts)
for evidence lookup, but are not graph nodes in the exports (they would
swamp a visualisation).
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from typing import Any, Optional

import networkx as nx


def entity_id(entity_type: str, canonical: str) -> str:
    return f"{entity_type}:{canonical}"


@dataclass
class KnowledgeGraph:
    run_id: str
    query_id: Optional[str]
    papers: dict[str, dict] = field(default_factory=dict)        # paper_id -> {title, year, relevance, on_topic, ...}
    entities: dict[str, dict] = field(default_factory=dict)      # entity_id -> {type, name, aliases, papers, chunks, community}
    mentions: dict[str, dict[str, int]] = field(default_factory=dict)  # chunk_id -> {entity_id: count}
    chunk_meta: dict[str, dict] = field(default_factory=dict)    # chunk_id -> {paper_id, section, section_type, page_start, page_end}
    cooccurrence: list[dict] = field(default_factory=list)       # {a, b, papers, pmi}
    statements: list[dict] = field(default_factory=list)         # gap statements
    result_claims: list[dict] = field(default_factory=list)      # directional method/task result claims
    communities: list[dict] = field(default_factory=list)        # {id, label, entities, size}
    stats: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_payload(self) -> dict:
        return {
            "run_id": self.run_id,
            "query_id": self.query_id,
            "papers": self.papers,
            "entities": self.entities,
            "mentions": self.mentions,
            "chunk_meta": self.chunk_meta,
            "cooccurrence": self.cooccurrence,
            "statements": self.statements,
            "result_claims": self.result_claims,
            "communities": self.communities,
            "stats": self.stats,
            "params": self.params,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "KnowledgeGraph":
        values = {k: payload.get(k) for k in (
            "run_id", "query_id", "papers", "entities", "mentions", "chunk_meta",
            "cooccurrence", "statements", "result_claims", "communities", "stats", "params",
        )}
        for key in ("papers", "entities", "mentions", "chunk_meta", "cooccurrence", "statements",
                    "result_claims", "communities", "stats", "params"):
            if values[key] is None:
                values[key] = {}
        for key in ("cooccurrence", "statements", "result_claims", "communities"):
            values[key] = values[key] or []
        return cls(**values)

    # ------------------------------------------------------------------ #
    # Views
    # ------------------------------------------------------------------ #

    @property
    def on_topic_papers(self) -> list[str]:
        return [pid for pid, p in self.papers.items() if p.get("on_topic", True)]

    def cooccurrence_graph(self, min_papers: int = 1) -> nx.Graph:
        g = nx.Graph()
        for eid, e in self.entities.items():
            g.add_node(eid, type=e["type"], name=e["name"], papers=len(e["papers"]))
        for edge in self.cooccurrence:
            if edge["papers"] >= min_papers:
                g.add_edge(edge["a"], edge["b"], weight=edge["papers"], pmi=edge["pmi"])
        return g

    def entity_papers(self, eid: str) -> set[str]:
        return set(self.entities.get(eid, {}).get("papers", []))

    def chunks_mentioning(self, eids: list[str]) -> list[str]:
        """Chunk ids mentioning all of ``eids`` (most mentions first)."""
        scored = []
        for cid, counts in self.mentions.items():
            if all(e in counts for e in eids):
                scored.append((sum(counts[e] for e in eids), cid))
        return [cid for _, cid in sorted(scored, reverse=True)]

    def to_networkx(self, cooccurrence_min_papers: int = 2) -> nx.MultiDiGraph:
        """Readable typed graph (papers, entities, statements, claims, communities)."""
        g = nx.MultiDiGraph()
        for pid, p in self.papers.items():
            if not p.get("on_topic", True):
                continue
            g.add_node(f"paper:{pid}", kind="Paper", label=(p.get("title") or pid)[:80],
                       year=p.get("year") or 0, relevance=round(p.get("relevance", 0.0), 3))
        for eid, e in self.entities.items():
            g.add_node(eid, kind="Entity", label=e["name"], entity_type=e["type"],
                       papers=len(e["papers"]), community=e.get("community", -1))
        for c in self.communities:
            g.add_node(f"community:{c['id']}", kind="Community", label=c["label"], size=c["size"])
            for eid in c["entities"]:
                if eid in self.entities:
                    g.add_edge(eid, f"community:{c['id']}", kind="IN_COMMUNITY")
        paper_entity: dict[tuple[str, str], int] = {}
        for cid, counts in self.mentions.items():
            pid = self.chunk_meta.get(cid, {}).get("paper_id")
            if pid is None or not self.papers.get(pid, {}).get("on_topic", True):
                continue
            for eid in counts:
                paper_entity[(pid, eid)] = paper_entity.get((pid, eid), 0) + 1
        for (pid, eid), n in paper_entity.items():
            g.add_edge(f"paper:{pid}", eid, kind="MENTIONS", weight=n)
        for edge in self.cooccurrence:
            if edge["papers"] >= cooccurrence_min_papers:
                g.add_edge(edge["a"], edge["b"], kind="CO_OCCURS", weight=edge["papers"], pmi=round(edge["pmi"], 3))
        for s in self.statements:
            sid = f"statement:{s['id']}"
            g.add_node(sid, kind="GapStatement", label=s["sentence"][:120], cue_class=s["cue_class"],
                       strength=s["strength"])
            g.add_edge(f"paper:{s['paper_id']}", sid, kind="STATES")
            for eid in s["entities"]:
                g.add_edge(sid, eid, kind="ABOUT")
        for i, claim in enumerate(self.result_claims):
            cid = f"result_claim:{i}"
            g.add_node(cid, kind="ResultClaim", label=claim["sentence"][:120], direction=claim["direction"],
                       metric=(claim.get("metric") or {}).get("name", ""),
                       metric_value=(claim.get("metric") or {}).get("value", ""))
            pid = claim.get("paper_id")
            if pid and f"paper:{pid}" in g:
                g.add_edge(f"paper:{pid}", cid, kind="REPORTS")
            for eid in claim["entities"]:
                if eid in self.entities:
                    g.add_edge(cid, eid, kind="ON")
        return g

    def to_cytoscape(self, cooccurrence_min_papers: int = 2) -> dict:
        g = self.to_networkx(cooccurrence_min_papers)
        nodes = [{"data": {"id": n, **attrs}} for n, attrs in g.nodes(data=True)]
        edges = [
            {"data": {"id": f"e{i}", "source": u, "target": v, **attrs}}
            for i, (u, v, attrs) in enumerate(g.edges(data=True))
        ]
        return {"elements": {"nodes": nodes, "edges": edges}}

    def to_graphml(self, cooccurrence_min_papers: int = 2) -> str:
        g = self.to_networkx(cooccurrence_min_papers)
        buffer = io.BytesIO()
        nx.write_graphml(g, buffer)
        return buffer.getvalue().decode("utf-8")


def build_cooccurrence(entities: dict[str, dict], n_papers: int) -> list[dict]:
    """Paper-level co-occurrence edges with PMI (log base 2)."""
    by_paper: dict[str, set[str]] = {}
    for eid, e in entities.items():
        for pid in e["papers"]:
            by_paper.setdefault(pid, set()).add(eid)
    pair_counts: dict[tuple[str, str], int] = {}
    for eids in by_paper.values():
        ordered = sorted(eids)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1:]:
                pair_counts[(a, b)] = pair_counts.get((a, b), 0) + 1
    edges = []
    n = max(1, n_papers)
    for (a, b), n_ab in pair_counts.items():
        n_a, n_b = len(entities[a]["papers"]), len(entities[b]["papers"])
        pmi = math.log2((n_ab * n) / (n_a * n_b)) if n_a and n_b else 0.0
        edges.append({"a": a, "b": b, "papers": n_ab, "pmi": round(pmi, 4)})
    return edges


def louvain_communities(entities: dict[str, dict], cooccurrence: list[dict], seed: int = 42,
                        min_papers: int = 2) -> list[dict[str, Any]]:
    """Louvain communities over the co-occurrence graph (edges with >= min_papers)."""
    g = nx.Graph()
    g.add_nodes_from(entities)
    for edge in cooccurrence:
        if edge["papers"] >= min_papers and edge["pmi"] > 0:
            g.add_edge(edge["a"], edge["b"], weight=edge["papers"] * edge["pmi"])
    if g.number_of_edges() == 0:
        return []
    groups = nx.community.louvain_communities(g, weight="weight", seed=seed)
    out = []
    for i, members in enumerate(sorted(groups, key=lambda m: -len(m))):
        members = sorted(members, key=lambda e: -len(entities[e]["papers"]))
        if len(members) < 2:
            continue
        label = ", ".join(entities[e]["name"] for e in members[:3])
        out.append({"id": len(out), "label": label, "entities": members, "size": len(members)})
    return out
