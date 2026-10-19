# How research gaps are found: two papers, 15 tools, one methodology

Sources:
- the two uploaded papers, read in full;
- the 15 tools, described from their public documentation and behaviour as of
  mid-2026. I did not re-check them online today, so the details of a tool's
  features may have moved.

The last section maps everything onto your pipeline (EWRGV) and detectors D1–D5.

---

## 1. The two papers

### ResearchRAG (Ghosh, 2026, preprint)

**Pipeline:**
- SciSpaCy entities (method, dataset, task, metric, model) per 256-token chunk;
- canonicalised against the CS Ontology and UMLS;
- a Neo4j knowledge graph with co-occurrence, citation and authorship edges;
- hybrid retrieval: MiniLM + FAISS, BM25, RRF with k=60, then a cross-encoder rerank;
- two-hop knowledge-graph expansion.

**How it finds gaps (GNN-RGS):**
1. Group entities into **concept clusters** in the graph.
2. A Graph Attention Network, trained on 1,200 hand-labelled clusters,
   predicts **knowledge density ρ̂ ∈ [0,1]** per cluster.
3. Score: `GapScore = 0.45·(1 − ρ̂) + 0.35·τ (temporal staleness) + 0.20·η (novelty)`.
4. The top sparse, stale clusters are written up by **contrastive LLM prompting**.
5. A gap card shows the score breakdown, supporting passages, a
   publication-year histogram and a suggested follow-up query.

**Evaluation:**
- 250 gap queries rated 1–5 by two PhD annotators on Relevance, Specificity
  and Novelty (κ=0.67);
- Spearman ρ=0.71 against expert ranking, vs 0.48 for an edge-count heuristic;
- removing GNN-RGS drops the composite score from 4.04 to 3.41.

**Failure modes it reports (directly relevant to you):**
- **24% spurious gaps from corpus incompleteness.** Fields with fewer than 30
  indexed papers produced fake gaps. Your current run has 32 papers with full text.
- **18% generic statements** when all clusters get similar density, which
  happens on broad questions.
- Two-hop knowledge-graph expansion pulled in off-topic entities
  ("contrastive loss" linked text work to image papers).

**Caveats:**
- τ and η are never given a formula.
- It's a single-author preprint with very strong claims.

Cite the GapScore *form* and the error analysis, not its numbers.

**What to take:**
- density + staleness + novelty as separate, explainable signals;
- the score breakdown on every gap;
- the corpus-adequacy warning.

The GNN needs 1,200 labelled clusters you don't have, so use the
**heuristic density** (their 0.48 baseline) or co-occurrence density instead.

### HySemRAG (Godinez, 2025, preprint)

**Pipeline:**
- metadata from several sources and asynchronous PDF download;
- Docling layout analysis;
- **LLM structured field extraction** (Qwen3-32B, 8k-token chunks, filling
  configured fields per paper);
- **LDA topic modelling**, with grid search on Cv coherence and LLM-written topic labels;
- **semantic unification** (MiniLM embeddings against a synonym dictionary;
  a cosine ≥ 0.55 maps a term like "no-till / zero tillage" to one canonical
  term, otherwise it is rejected);
- a Neo4j graph (Article → Method / Disease / StudyType …);
- Qdrant vectors;
- three-way RRF (vector, keyword, graph);
- a generator + QA-agent loop and post-hoc citation verification.

**How it finds gaps:** it doesn't score them. It **counts**. Graph queries
aggregate which methods, exposures and study types appear across papers. Its
ozone example: Random Forest in 20 of 34 studies, GBM 14, SVM 13 and so on.
Under-represented methods, combinations and study types are read off as
methodological gaps. LDA topic prevalence shows under-covered themes.

**Evaluation:**
- structured fields beat raw PDF chunks for retrieval (0.655 vs 0.485 similarity);
- QA single-pass success 68%;
- citation accuracy 99% after verification.

**Gap quality isn't measured**, and performance drifted by −1.9% a day.

**What to take:**
- **structured fields per paper** (method, dataset, metric, population, study type);
- **semantic unification of terms before counting**;
- method-distribution tables as the evidence for a methodological gap.

---

## 2. The 15 tools: what each does that's relevant to gaps

| Tool | Gap-relevant mechanism | Gap type it supports | Reusable? |
|---|---|---|---|
| **Elicit** | LLM extracts columns per paper (population, intervention, outcome, methods, **limitations**) into a table; you scan for empty or weak cells | explicit-limitation, combination | Yes: an evidence table is D1 + D2 |
| **Scite** | Smart Citations label each citing sentence **supporting / contrasting / mentioning** | contradictory evidence; counter-evidence for validation | Yes: citation-stance check |
| **Consensus** | Extracts each paper's claim on a yes/no question and aggregates a **Consensus Meter** (yes / possibly / no) | disagreement and thin evidence | Yes: stance aggregation |
| **ResearchRabbit** | Citation-network exploration (earlier, later, similar work, author networks) | under-connected areas | Partly: network structure |
| **Semantic Scholar** | TLDRs, **citation intent** (background, method, result), influential citations, SPECTER embeddings, recommendations API | which work builds on which | Yes: free API you already use |
| **SciSpace** | Chat-with-PDF and literature-review tables with **limitations / future work** columns | explicit-limitation | Yes: same as D1 |
| **Connected Papers** | Similarity graph from **co-citation + bibliographic coupling**; prior and derivative works | sparse regions, bridging gaps | Partly: needs citation data |
| **Litmaps** | Citation maps on a **time axis** from seed papers; monitoring new work | temporal gaps, stale lines | Yes: year × topic |
| **Google Scholar** | Search + "cited by" counts; no gap features | manual only | No |
| **OpenAlex** | **Topics / concepts taxonomy**, works counts by year, citation links, free API | density and trend per topic | Yes: you already collect from it |
| **Perplexity** | Web-search LLM answers with citations | none structured | No (LLM-only) |
| **NotebookLM** | LLM grounded strictly on uploaded sources, cited answers | explicit gaps you ask for | Only as a grounding pattern |
| **Dimensions** | Publications + **grants + patents + clinical trials** analytics; topic trends | funding vs output mismatch | Out of scope (paid) |
| **Web of Science** | Citation reports, Analyze Results by year and research area, ESI hot or highly cited papers | density, trend, emerging fronts | Pattern only (paid) |
| **Scopus / SciVal** | Analyze search results; **Topic Prominence** (momentum of topic clusters) | emerging vs stagnant topics | Pattern only (paid) |

**What none of them do:** none of the tools *validates* that a gap is real.
They surface candidates:
- Elicit, SciSpace and NotebookLM through **LLM extraction and summaries**;
- OpenAlex, Dimensions, WoS and Scopus through **counts**;
- Connected Papers, ResearchRabbit and Litmaps through **networks**;
- Scite and Consensus through **stance**.

The researcher still judges. That missing validation step is the space your
EWRGV contribution sits in.

---

## 3. The methodology, extracted

Eight distinct ways of finding a gap appear across the papers and tools:

| # | Method | Signal | Seen in |
|---|---|---|---|
| M1 | **Explicit-statement mining** | authors say "remains unexplored", "future work", "limited to" in Limitations, Future Work, Conclusion or Research Gap sections | Elicit, SciSpace, NotebookLM |
| M2 | **Structured evidence matrix** | per-paper fields (method, dataset, metric, population, study type) → empty or thin cells in method × dataset, method × metric | Elicit tables, HySemRAG fields + graph counts, ResearchRAG co-occurrence |
| M3 | **Term unification** (pre-step for M2) | merge synonyms by embedding similarity before counting | HySemRAG semantic unification, ResearchRAG CSO/UMLS |
| M4 | **Density** | few papers or chunks per concept cluster | ResearchRAG ρ̂, OpenAlex, WoS, Scopus counts |
| M5 | **Temporal staleness / momentum** | a concept's activity stopped or declined | ResearchRAG τ, Litmaps, SciVal prominence, Dimensions trends |
| M6 | **Novelty / topic coverage** | a cluster far from established clusters, or a low-prevalence LDA topic | ResearchRAG η, HySemRAG LDA |
| M7 | **Network structure** | weakly connected clusters, missing bridges, isolated lines of work | Connected Papers, ResearchRabbit, Litmaps |
| M8 | **Stance / contradiction** | papers disagree, or contrasting citations | Scite, Consensus |

Plus two **guards**, both learned from failures:
- **G1 Corpus adequacy:** don't trust density gaps when the corpus is small
  (ResearchRAG: fewer than 30 papers gives 24% spurious gaps). Report coverage
  next to every density-based gap.
- **G1b External coverage check (your validation):** before calling a gap
  genuine, search the wider literature (OpenAlex, Semantic Scholar) with
  direct, synonym and method-variant queries, and count what already fills it.
  This is what ResearchRAG lacks, and it causes its 24% spurious gaps.
- **G2 Grounded wording:** an LLM may only *word* a gap from attached evidence,
  with each sentence checked against passages (ResearchRAG CSCG, HySemRAG QA
  agent, NotebookLM). It never decides that a gap exists. For example, a sentence whose best
  passage similarity is below 0.62 gets re-retrieved, or marked [UNCERTAIN].

---

## 4. Mapped onto your pipeline

| Method | Your detector | Data you already have | Status |
|---|---|---|---|
| M1 explicit statements | **D1** | section-typed chunks, `gap_evidence` filter, bge embeddings to group across papers | next to build |
| M2 + M3 matrix with unification | **D2** | query understanding (methods, datasets, metrics) as the dictionary; bge for synonym merging | after D1 |
| M4 + M5 + M6 density, staleness, novelty | **D3 + D4** | publication years, chunk embeddings (HDBSCAN), OpenAlex topic counts by year | later |
| M7 network | optional | OpenAlex and Semantic Scholar citation links | optional, later |
| M8 stance / contradiction | **D5** + EWRGV validation | multi-query retrieval, Methods and Results chunks; Semantic Scholar citation intent | after D1–D2 |
| G1 corpus adequacy | coverage check | paper count per concept, acquisition yield (24 of 141 PDFs) | add with D3 |
| G2 grounded wording | LLM baseline / wording step | your friend's grounding check | exists |

**Proposed scoring** (evidence-weighted, no LLM confidence):

```
GapScore = w1·explicit   (D1: papers stating it, section quality)
         + w2·(1 − density)  (D2/D4: observed/expected co-occurrence)
         + w3·staleness      (D3)
         + w4·novelty        (D4)
         − w5·counter        (D5: papers that already address it)
reported with: coverage (G1), evidence passages, year histogram
```

Start with ResearchRAG's α/β/γ split for density, staleness and novelty, add
the explicit and counter terms, and tune the weights on a small hand-rated
set. Evaluate the way ResearchRAG does: Likert 1–5 on relevance, specificity
and novelty from two raters, plus Spearman ρ against their ranking. Compare
against your friend's LLM detector as Baseline 1.

### What your own baseline run already showed

The LLM detector on run `e0b54bcc` cited one paper per gap and missed the
"Research Gap" and "Conclusion" sections. That's exactly what M1 fixes. It also
inverted three claims, which is what M8 catches. And with only 32 full-text
papers, G1 says density-based gaps on this run need a coverage warning.
