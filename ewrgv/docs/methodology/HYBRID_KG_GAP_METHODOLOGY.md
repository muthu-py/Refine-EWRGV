# EWRGV-KG: Hybrid Knowledge-Graph Gap Detection with Evidence-Weighted Validation

Status: **P1–P4 implemented (2026-10-19)**: relevance gate, entity tagging, KG + Louvain communities, gap statements and directional result claims; D1–D4, fusion, GapScore, grounding; and EWRGV corpus/external literature validation with persisted results and `/kg`, `/gaps`, `/validate` APIs. P5–P6 remain in progress.

Implementation notes vs this design: the KG is persisted as one JSONB payload per run (`kg_graphs`) rather than separate tables; the relevance gate is run-relative (median − 0.08, floor 0.55) because bge scores every retrieved paper 0.6–0.9; synonym merging uses cosine 0.88 and never merges two curated lexicon entities; sparsity is discounted by min(1, N/50) for small corpora.
Prepared 2026-10-19. Synthesises:

- **ResearchRAG** (Ghosh, 2026): an academic knowledge graph (SciSpaCy entities, co-occurrence, citation and author edges); the gap score
  `α(1−ρ) + βτ + γη`; citation-aware self-correction (θ = 0.62); and its error analysis, which found **24% spurious gaps caused by corpus incompleteness**.
- **HySemRAG** (Godinez, 2025): per-paper structured field extraction; semantic unification (embedding similarity ≥ 0.55); a knowledge graph plus LDA topic prevalence; gaps read from method distributions; generator and evaluator QA.
- **Tools** (Elicit, SciSpace, Scite, Consensus, ResearchRabbit, Connected Papers, Litmaps, Semantic Scholar, OpenAlex, Scopus, Web of Science, Dimensions, Google Scholar, Perplexity, NotebookLM). Each tool exposes one gap *signal*: extraction tables, network sparsity, citation stance, bibliometric trend, or LLM synthesis. None scores or validates gaps.
- **This project's findings so far.** The baseline that uses only an LLM detector, run on run `e0b54bcc`, showed four failure modes:
  - every gap rested on 1 passage from 1 paper;
  - it inverted claims (for example, "need uncertainty estimates", citing a paper that already provides them);
  - it ignored explicit gap sections;
  - it cited no counter-evidence (0 contradicting passages).

**Core idea.** Detection is driven by a **knowledge graph plus measurable signals**. Each candidate gap is a *hypothesis*, and every number behind it is recorded. **EWRGV validation** then tests each hypothesis against the run corpus **and the external literature**. That validation step is what ResearchRAG lacks, and its absence causes ResearchRAG's spurious gaps. Signal detection is the default. An LLM can optionally propose additional candidates grounded in retrieved corpus passages or refine the wording of signal candidates; its self-reported confidence is metadata, never evidence or part of a signal gap score. Every candidate remains subject to EWRGV validation.

---

## 1. Architecture

```
L0  Corpus (implemented)
    /query (stored) → /collect (question + all expansions) → acquire → parse → sections
    → chunks → represent (bge-small 384-d + BM25) → retrieval (hybrid, multi-query RRF,
    gap_evidence filter, noise filter, per-paper cap, corpus cache)
        │
L0b Relevance gate (new): drop off-topic papers (crop disease, cardiology, …)
        │
L1  Structured extraction (HySemRAG + ResearchRAG)
    entities · canonicalisation · explicit gap statements · result claims · per-paper record
        │
L2  Knowledge graph  (ResearchRAG AKG, in-process: networkx + Postgres tables)
    Paper · Chunk · Entity{METHOD,DATASET,TASK,METRIC,MODALITY,DISEASE,POPULATION}
    · GapStatement · ResultClaim · Topic   +  edges MENTIONS / CO_OCCURS / STATES / ABOUT /
    REPORTS / IN_TOPIC / CITES / SIMILAR_TO
        │
L3  Candidate generation — six detectors over the KG
    D1 explicit statements · D2 combination sparsity · D3 temporal staleness
    D4 topic density/novelty · D5 contradiction · D6 LLM detector (baseline, optional)
        │
L4  Fusion + GapScore (defined τ, η, ρ; agreement bonus; grounding gate)
        │
L5  Structured gap representation (from KG entities; LLM optional for wording)
        │
L6  EWRGV validation (the contribution)
    verification queries (KG-expanded) → direct / terminology / method / counter searches
    on run corpus + OpenAlex + Semantic Scholar → evidence classification → coverage
    (bibliometric expected-vs-observed) → aggregation → confidence → classification
    → evidence-chain explanation (+ self-correction for ungrounded sentences)
        │
L7  API + report  (/kg, /gaps, /validate; gap cards; KG export for visualisation)
```

The knowledge graph is the hub of the design. Detectors D2 to D5 are queries over it. Validation uses its neighbourhoods (synonyms, sibling methods, sibling datasets) to build verification queries. The report renders the subgraph behind each gap.

> `docs/architecture/ARCHITECTURE.md` currently lists knowledge graphs under "Explicitly NOT used", so that line must be updated. This design keeps that document's intent: no Neo4j server, no GNN training, no GraphRAG infrastructure. The graph is built in process (`networkx`) and persisted in Postgres.

---

## 2. L0b — Corpus relevance gate

**Problem.** `/collect` pulls in off-topic papers. Run `e0b54bcc` contains, for example, a crop-disease paper and a cardiovascular-risk paper.

**Method.**
1. Compute `rel(p) = cos(embed(question + expansions centroid), embed(title + abstract of p))`.
2. Keep papers with `rel ≥ θ_rel`. The default is 0.45, calibrated on the run.
3. Below that threshold, mark the paper `OFF_TOPIC`. It stays in the run and is reported, but detection skips it.

**Why.** This has the same effect as the human screening step in Elicit and in systematic reviews, and it stops spurious gaps from leaking out of unrelated literature.

---

## 3. L1 — Structured extraction

### 3.1 Entity tagging (ResearchRAG NER + HySemRAG fields)

| Entity type | Sources, cheapest first |
|---|---|
| METHOD, TASK | the stored query understanding (`methods`, `method_variants`, `task`, `concepts`); a curated domain list (CNN, ViT, U-Net, transformer, SVM, …); SciSpaCy `en_core_sci_md`, optional |
| DATASET | a regex for dataset idioms ("X dataset", "the X database", capitalised and acronym names near "dataset/images/scans"); a known list (APTOS, EyePACS, Messidor, DRIVE, Kermany OCT, ODIR, RFMiD, …) |
| METRIC | a fixed lexicon (accuracy, AUC, F1, sensitivity, specificity, kappa, Dice, IoU, …) with number capture |
| MODALITY | fundus, OCT, OCTA, UWF, slit-lamp, FFA, … |
| DISEASE / POPULATION | the query understanding's `entities` and `constraints`; SciSpaCy `DISEASE`; demographic and geographic cues |

Every mention records `(chunk_id, entity, span, section_type)`.

### 3.2 Canonicalisation (HySemRAG semantic unification)

- Embed each surface form with the existing bge model.
- Assign the form to the nearest canonical entity of the same type if `cos ≥ θ_unify`. The default is 0.82; bge similarities run higher than the MiniLM similarities behind HySemRAG's 0.55, so the threshold is calibrated on a labelled sample.
- Otherwise the form becomes a new canonical entity.
- An alias table keeps every surface form. These aliases become the **synonyms** used by validation.

### 3.3 Explicit gap statements (SciSpace "Limitations / Future work" columns, made automatic)

- **Scope:** sentences in the LIMITATIONS, FUTURE_WORK, CONCLUSION and DISCUSSION sections, and in any section whose heading matches "research gap|challenge|open problem".
- **Cue lexicon**, in three classes:
  - **LIMITATION:** "limited to", "only (tested|evaluated) on", "small (sample|dataset)", "did not", "lack of", "fails to"
  - **FUTURE_WORK:** "future (work|research|studies) (should|could|will)", "remains to be"
  - **OPEN_PROBLEM:** "remains (unexplored|unresolved|an open)", "has not been (studied|addressed)", "few studies", "challenge(s) remain"
- **Each statement stores:**
  - the cue class;
  - its strength: explicit "has not been studied" counts 1.0; a hedged "could be explored" counts 0.6;
  - its entities (from 3.1);
  - its embedding;
  - its paper, year and section.
- **Self-reference filter.** Drop statements about the paper's own pipeline that do not generalise, for example "our app's UI…".

### 3.4 Result claims (input to D5, contradiction)

Sentences in the RESULTS, DISCUSSION and ABSTRACT sections that contain a METHOD plus a DATASET/TASK plus a direction cue. Direction is one of:

- **POSITIVE:** "outperforms", "improves", "achieves higher"
- **NEGATIVE:** "fails", "worse", "does not improve", "no significant"
- **MIXED**

Numeric metrics are captured where present.

### 3.5 Per-paper record (HySemRAG structured fields, optional LLM)

An optional pass: an LLM fills one record per paper (methods, datasets, modality, task, metrics, limitations), using the chunk text **only as grounding**. Every field must quote a supporting span, otherwise the field is dropped. HySemRAG showed structured fields retrieve better than raw chunks (0.655 vs 0.485). The record is merged into the graph with `source = llm_extracted`, kept separate from rule- and NER-tagged facts so the two can be ablated.

---

## 4. L2 — Knowledge graph

**Nodes**

| Node | Key attributes |
|---|---|
| Paper | year, venue, authors, citation count, relevance, OFF_TOPIC flag |
| Chunk | section_type, page |
| Entity | type, canonical name, aliases, document frequency |
| GapStatement | cue class, strength, text |
| ResultClaim | direction, metric |
| Topic | cluster id, size, centroid, label |

**Edges**

| Edge | Meaning, weight |
|---|---|
| Paper –HAS→ Chunk | |
| Chunk –MENTIONS→ Entity | count |
| Entity –CO_OCCURS→ Entity | **paper-level** co-occurrence count and PMI |
| Paper –STATES→ GapStatement | |
| GapStatement –ABOUT→ Entity | |
| Paper –REPORTS→ ResultClaim –ON→ Entity | |
| Chunk –IN_TOPIC→ Topic | |
| Paper –CITES→ Paper | from OpenAlex `referenced_works` or Semantic Scholar references, when both papers are in the run |
| Entity –SIMILAR_TO→ Entity | embedding similarity in [0.70, θ_unify): siblings, not synonyms |

**Storage**
- Postgres tables `kg_entities`, `kg_aliases`, `kg_mentions`, `kg_edges`, `gap_statements`, `result_claims`, `kg_topics` (migration 10).
- The graph is rebuilt per run into `networkx` for analysis.
- **Export:** JSON (Cytoscape/Sigma) and GraphML, so the report shows the subgraph behind each gap, as in ResearchRAG's gap panel.

**Build cost:** seconds for runs of 30 to 1,000 papers, with no server, unlike ResearchRAG's 4.2 hours on Neo4j.

---

## 5. L3 — Six detectors

Every detector emits `CandidateGap` objects with `metadata.signals = {raw numbers}`, `metadata.detector` and the evidence chunk ids.

### D1 — Explicit-statement consensus (SciSpace, Elicit; HySemRAG fields)

1. Cluster the GapStatements by embedding, using agglomerative clustering with average linkage and a distance threshold of 1 − 0.72.
2. A cluster becomes one candidate.

Its signals:
- **support** = Σ strength;
- **papers** = number of distinct papers;
- **section mix**;
- **latest year**;
- **addressed_later** = whether a newer paper in the corpus mentions the same entities in METHODS/RESULTS, which is early counter-evidence.

The gap type comes from the entity types involved:

| Entities involved | Gap type |
|---|---|
| DATASET | Dataset |
| METRIC or an evaluation cue | Evaluation |
| POPULATION or geographic | Population / Geographic |
| otherwise | Methodological |

*On run `e0b54bcc` this surfaces gaps such as "generalisation across datasets and modalities" (≥ 3 papers), "class imbalance", "dependence on labelled data" and "single-disease rather than multi-disease screening". The LLM-only baseline missed or under-counted all of these.*

### D2 — Combination sparsity on the KG (HySemRAG distributions, ResearchRAG density)

For each type pair `(A, B)` in {METHOD×DATASET, METHOD×MODALITY, METHOD×TASK, TASK×POPULATION, METHOD×METRIC}:

- `n_A`, `n_B` = paper counts; `N` = papers in the run (excluding OFF_TOPIC)
- `observed = n_AB`, `expected = n_A·n_B / N`
- `ρ_AB = observed / expected` (lift), clipped to [0, 1] for gap scoring
- candidates are the cells with `n_A ≥ m`, `n_B ≥ m` (default m = 3) and `ρ_AB ≤ 0.25`

The statement template reads: *"{A} is common (n_A papers) and {B} is common (n_B), but they are combined in only {n_AB} paper(s)."*

**This is a hypothesis only.** The run corpus is small, so D2 candidates *must* pass external coverage (L6). That is ResearchRAG's 24% failure mode, handled explicitly.

### D3 — Temporal staleness (ResearchRAG τ, defined; Litmaps, OpenAlex trends)

For an entity, statement cluster or topic `x`:

- `Y_ref` = latest year in the corpus;
- `Y_last(x)` = latest year in which `x` appears;
- `share_t(x)` = papers on `x` in year *t* / all papers in year *t*;
- `slope(x)` = least-squares slope of `share_t` over the last 5 years.

Staleness:

    τ(x) = 0.6 · min(1, (Y_ref − Y_last(x)) / H) + 0.4 · σ(−k · slope(x))      H = 5 years, k = 20

Plus a **stale future-work** rule: a D1 cluster whose newest statement is ≥ 3 years old and has no `addressed_later` evidence gets τ ≥ 0.8. On short corpora (all papers within 2 to 3 years, as here) τ is reported but down-weighted automatically (see §6).

### D4 — Topic density and novelty (ResearchRAG ρ and η; HySemRAG LDA; Connected Papers clusters)

1. Cluster the chunk embeddings of non-noise, on-topic chunks with HDBSCAN (`min_cluster_size = 5`). LDA is the alternative for the HySemRAG ablation.
2. Label each topic by its top c-TF-IDF terms, with an optional LLM label.

Then:
- **density** `ρ(T)` = papers(T) / median papers per topic, clipped to [0, 1]. This is the *heuristic edge-count density*, the ρ = 0.48 baseline in ResearchRAG; a learned GNN is out of scope.
- **novelty** `η(T)` = 0.5 · (1 − max cosine of T's centroid to the size-weighted centroids of large topics) + 0.5 · (share of T's papers from the last 2 years).
- **candidates:** topics with `ρ ≤ 0.35`, relevance to the question ≥ θ_rel, and support from at least 2 papers.

### D5 — Contradiction (Scite contrasting, Consensus meter)

1. Group the ResultClaims by (METHOD, DATASET or TASK).
2. A group with both POSITIVE and NEGATIVE claims from **different** papers becomes a *Contradictory Evidence* candidate.
3. Optional confirmation: an NLI cross-encoder (`cross-encoder/nli-deberta-v3-base`) labels claim pairs as contradiction with probability ≥ 0.7.

Signal: `contradiction = min(n_pos, n_neg) / (n_pos + n_neg)`, together with the number of papers on each side.

### D6 — LLM detector (the existing `GapDetector`, optional)

The existing detector, kept as **Baseline 1** for comparison and available as an opt-in proposal source in `/gaps`. Proposal candidates are labeled with their source and cited retrieved corpus chunks. Near-duplicates merge with signal candidates while retaining the LLM proposal metadata. Its `initial_confidence` is stored and **never scored**. The standalone baseline can also be stored separately for `view=compare`.

---

## 6. L4 — Fusion and GapScore

### 6.1 Merge

Two candidates merge when either condition holds:
- the statement embeddings have `cos ≥ 0.80`;
- the entity sets have Jaccard ≥ 0.6 and the gap type is the same.

The merged candidate keeps the union of evidence and records `detectors = {D1, D2, …}`.

### 6.2 Signals, all in [0, 1]

| Signal | Definition | Origin |
|---|---|---|
| `E` explicit | 1 − exp(−Σ strength / 2) from D1 statements on the candidate | SciSpace, Elicit |
| `S` sparsity | 1 − ρ (D2 lift or D4 density; the minimum if both) | ResearchRAG α-term |
| `τ` staleness | D3 | ResearchRAG β-term |
| `η` novelty | D4 | ResearchRAG γ-term |
| `C` contradiction | D5 | Scite, Consensus |
| `I` independence | min(distinct papers / 3, 1) · venue-diversity factor | evidence independence |
| `A` agreement | (number of detectors − 1) / 3, clipped | hybrid agreement |

### 6.3 Score

    GapScore = 0.25·E + 0.25·S + 0.15·τ + 0.10·η + 0.10·C + 0.15·I   (+ 0.10·A bonus, capped at 1)

- Weights renormalise over the **available** signals. τ's weight halves automatically when the corpus spans fewer than 4 years.
- The **ResearchRAG form** `0.45(1−ρ) + 0.35τ + 0.20η` is computed alongside as a published baseline.
- Weights live in config, and are tuned only on the annotated development split (§9).

### 6.4 Grounding gate (ResearchRAG self-correction)

- Every candidate needs at least one evidence chunk with `cos(statement, chunk) ≥ 0.60`.
- If none qualifies, re-retrieve with the statement's key noun phrases, up to 3 times, using hybrid multi-query retrieval over the gap-evidence sections.
- If it is still ungrounded, mark it `UNCERTAIN` and exclude it from the top-K.

### 6.5 Ranking

Reuse `CandidateRanker`'s de-duplication and type-diversity logic, with `base = GapScore` and `confidence_weight = 0`. Take the top-K, default 10, into validation.

---

## 7. L5 — Structured gap representation

Built **deterministically from the graph**:

| Field | Source |
|---|---|
| domain | from the query understanding |
| population | POPULATION / DISEASE entities |
| methodology | METHOD |
| outcome | TASK / METRIC |
| temporal_scope | D3 |
| geographic_scope | geographic entities |
| key_concepts | the candidate's entities |
| synonyms | KG aliases plus `SIMILAR_TO` neighbours plus the query understanding's terminology and method variants |

An optional LLM rewrites `gap_statement` into fluent prose, constrained to the fields above. The original template statement is kept.

---

## 8. L6 — EWRGV validation (the contribution)

This implements the 11 steps of `METHODOLOGY.md` concretely.

**1. Verification queries.** Generated from the representation and its KG neighbourhood:

| Search | Query form | Example |
|---|---|---|
| direct | key concepts joined | "vision transformer" "OCT" "multi-disease" |
| terminology | each synonym or alias substitution | |
| method variation | substitute `SIMILAR_TO` sibling methods or datasets | "Swin transformer OCT multi-label" |
| counter | phrased as if the gap were filled | "{A} applied to {B}", "{A} on {B} dataset", "first … to …" |

**2–5. Searches.**
- The **run corpus**: hybrid retrieval with no section filter.
- **OpenAlex**: `search=` returns the hit count plus the top 25 records (title, abstract, year, concepts).
- **Semantic Scholar**: the same, top 25.
- Cached per query.

**6. Evidence classification**, per retrieved item:

| Evidence type | Rule |
|---|---|
| COUNTER | the item reports doing the gap: its entity set ⊇ the gap's core pair, in its method or results, or in an abstract with an action verb |
| PARTIAL | partial entity overlap, or the same task with a different modality |
| SUPPORTING | the item itself states the limitation (a D1 cue on the same entities) |
| CONTRADICTORY | the item's result directly contradicts the gap claim |
| INSUFFICIENT | otherwise |

Rules first. An NLI model can confirm COUNTER and SUPPORTING. An LLM is optional and must quote its evidence.

**7. Coverage analysis.** Bibliometric expected versus observed, taken from **external** counts:

    cov_ext = n_ext(A ∧ B) / max(1, min(n_ext(A), n_ext(B)))      (OpenAlex counts)
    cov_lit = #COUNTER(+0.5·#PARTIAL) among top-50 external items / 50
    cov_run = D2 lift in the run corpus

A gap that is sparse in the run but dense externally is a **false gap caused by an incomplete corpus**. This is exactly ResearchRAG's failure mode, now detected.

**8. Aggregation.** De-duplicate items by DOI or title. Weight each by recency (newer COUNTER evidence weighs more) and by independence.

**9. Confidence.**

    support = GapScore-evidence mass (E, I, A) + external SUPPORTING mass
    counter = w_c · (COUNTER + 0.5·PARTIAL) mass,  w_c = 1.2  (conservative)
    coverage_penalty = clip( (cov_ext − c0)/(c1 − c0), 0, 1 ),  c0 = 0.02, c1 = 0.15

    confidence = support/(support + counter) · (1 − 0.6 · coverage_penalty)

    If total evidence is below a minimum, cap confidence at 0.39 (INSUFFICIENT; "sparse ≠ gap").

**10. Classification.** Uses the existing thresholds:

| Confidence | Classification |
|---|---|
| ≥ 0.70 | VALID_GAP |
| 0.40–0.69 | UNCERTAIN_GAP |
| < 0.40 | UNSUPPORTED_GAP |

Each result also carries a *reason code*: `ADDRESSED_EXTERNALLY`, `CORPUS_ARTEFACT`, `CONTESTED` or `INSUFFICIENT_EVIDENCE`.

**11. Evidence-chain explanation.** A template narrative listing:
- which detectors fired, with their numbers;
- which papers state the gap, with quotes and sections;
- the external counts;
- the top counter items, with DOIs;
- the final arithmetic.

Every sentence is aligned to a source (cosine ≥ 0.62), otherwise it is marked [UNCERTAIN].

---

## 9. Evaluation (answers RQ1 and RQ4)

- **Gold set.** 40 to 60 candidate gaps from 2 or 3 runs, each labelled *genuine* or *spurious* by 2 annotators. They also rate relevance, specificity and novelty on a 1–5 Likert scale, and report Cohen's κ (ResearchRAG protocol).
- **Configurations:**
  1. LLM-only (D6)
  2. signals only (D1–D5)
  3. signals plus KG expansion
  4. 3 + LLM wording
  5. 3 + EWRGV validation (full)
  6. ResearchRAG-form score instead of GapScore
- **Metrics:**
  - precision of genuine gaps in the top-10;
  - **false-gap rate**, against the 24% reported by ResearchRAG;
  - Spearman ρ against the expert ranking (ResearchRAG reports 0.48 for its heuristic and 0.71 for its GNN);
  - Likert composite;
  - grounding rate.
- **RQ4.** Logistic regression of the genuine/spurious label on the individual signals (E, S, τ, η, C, I, A, cov_ext, counter mass), to show which signals separate genuine from false gaps.
- **Statistics:** Wilcoxon signed-rank tests with Bonferroni correction; bootstrap 95% confidence intervals.

---

## 10. API

| Method and path | Purpose |
|---|---|
| `POST /research/kg/build {run_id}` | background job: L0b + L1 + L2 |
| `GET /research/kg/{run_id}` | graph statistics; `?format=cytoscape|graphml` for export |
| `POST /research/gaps {run_id, query_id, detectors?, top_k}` | background job: L3–L5 |
| `GET /research/gaps/{run_id}` | ranked candidates with signals and evidence |
| `POST /research/validate {run_id, gap_ids?}` | background job: L6 |
| `GET /research/validate/{run_id}` | classifications, confidence and explanations |

---

## 11. Build plan (each phase testable on run `e0b54bcc`)

| Phase | Content | Depends on |
|---|---|---|
| **P1** | Relevance gate; entity tagging plus canonicalisation (query dictionary + lexicons + regex); KG tables (migration 10) and builder; `/kg/build`, `/kg/{run_id}` | done L0 |
| **P2** | Gap-statement and result-claim extraction; D1; GapScore v1 (E, I, A); grounding gate; `/gaps` | P1 |
| **P3** | D2 combination sparsity; D3 internal and OpenAlex staleness; D4 HDBSCAN topics; full GapScore plus the ResearchRAG baseline | implemented |
| **P4** | EWRGV validation: query generation, OpenAlex and Semantic Scholar search, evidence classification, coverage, confidence, classification, explanation; `/validate` and persisted results | implemented |
| **P5** | D5 contradiction (lexicon, then NLI); optional LLM per-paper records (§3.5); D6 merged as an ablation | P2 |
| **P6** | Gold set, ablations, RQ4 analysis, report export | P1–P5 |

**New dependencies:**
- `networkx`, and `scikit-learn`, which is already present and provides `HDBSCAN`;
- optional: `scispacy` + `en_core_sci_md` (P1 recall boost) and an NLI cross-encoder (P5).

No Neo4j, no GPU training, no paid API is required. The LLM is used only for optional wording, labels and records.
