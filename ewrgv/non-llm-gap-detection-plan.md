# Plan: non-LLM candidate gap detection, plus API and stub inventory

Prepared 2026-10-19 from the code on both branches and the documents you shared:
- the roadmap mind map
- 02 ResearchRAG Understanding
- 03 Literature Survey plan
- 04 Methodology Design Plan
- 05 Bundle
- the HySemRAG and ResearchRAG papers

No code was changed. "Teammate branch" means `feat/collection-backed-gap-detection` at `f2cd629`. "Main" means `origin/main` at `ba7b117` ("partial retrieval").

## 1. All APIs

Every route is mounted under `/api/v1`. The research routes have the `/research` prefix.

| Method and path | Main | Teammate branch | What it really does |
|---|---|---|---|
| `GET /health` | ✓ | ✓ | Works |
| `POST /research/query` | ✓ | ✓ | Works: LLM query understanding and expansion. Returns a stub if no API key is set. The result is not stored. |
| `POST /research/collect` | ✓ | ✓ | Works: Semantic Scholar and OpenAlex, then deduplication, then Supabase |
| `POST /research/acquire` | ✓ | ✓ | Works: open-access discovery and PDF download (background job) |
| `GET /research/{job_id}` | ✓ | ✓ | Works: acquisition job status |
| `POST /research/parse` | ✓ | ✗ | Works: PyMuPDF text extraction (background job) |
| `GET /research/parse/status/{job_id}` | ✓ | ✗ | Works |
| `POST /research/sections` | ✓ | ✗ | Works: heading detection and section types (Limitations, Future Work, …) |
| `GET /research/sections/status/{job_id}` | ✓ | ✗ | Works |
| `POST /research/chunks` | ✓ | ✗ | Works: section-aware chunking |
| `GET /research/chunks/status/{job_id}` | ✓ | ✗ | Works |
| `GET /research/corpus/health` | ✓ | ✗ | Works |
| `GET /research/corpus/health/inconsistencies` | ✓ | ✗ | Works |
| `POST /research/represent` | ✓ | ✗ | Works: embeddings plus the BM25 index build |
| `GET /research/represent/status/{job_id}` | ✓ | ✗ | Works |
| `POST /research/retrieve/dense` | ✗ | ✓ | **Demo only**: runs over the built-in stub corpus with hash "embeddings", not your papers |
| `POST /research/retrieve/sparse` | ✗ | ✓ | **Demo only**: BM25 over the stub corpus |
| `POST /research/retrieve/knowledge-graph` | ✗ | ✓ | **Demo only**: stub knowledge graph |
| `POST /research/retrieve/hybrid` | ✗ | ✓ | **Demo only**: RRF over the three above |
| `POST /research/gaps` | stub | stub | Returns `status: "not_implemented"` and an empty list on both branches |
| `POST /research/validate` | stub | stub | Returns `status: "not_implemented"` and an empty list on both branches |

Neither branch has an API that runs the full pipeline. `ResearchPipeline` is only built inside tests.

## 2. Stubs and unimplemented code

| Area | Main | Teammate branch |
|---|---|---|
| Retrieval over your corpus (`retrieval/bm25.py`, `semantic.py`, `fusion.py`) | Stub. The BM25 index is built, but nothing can query it. | Implemented, but only connected to the stub corpus (API) or to in-memory abstracts (pipeline) |
| Reranking (`retrieval/reranking.py`) | Stub (just truncates the list) | Stub (just truncates the list) |
| Real embeddings (`providers/embeddings/openai_embeddings.py`) | Implemented | Stub. The stubs use hash-based vectors. |
| Gap detection (`gap_detection/detector.py`, `ranking.py`, `representation.py`) | Stub | Implemented, LLM-based |
| Pre-validation evidence score (`gap_detection/evidence_scoring.py`) | Absent | Implemented |
| Pipeline stage 3 full-text use (`orchestration/research_pipeline.py`) | Stub | TODO: abstracts only, parser never called |
| Pipeline stage 5 (reranking) and stage 8 (validation) | Stub | TODO |
| Validation (`validation/validator.py`, `query_generation.py`, `direct_search.py`, `terminology_search.py`, `method_search.py`, `counter_search.py`, `iteration.py`) | Stub | Stub (unchanged) |
| Evidence classification and aggregation (`evidence/*`) | Stub | Stub |
| Coverage analysis (`coverage/*`) | Stub | Stub |
| Confidence (`scoring/confidence.py`, returns 0.0) | Stub | Stub |
| Explanation (`explanation/*`) | Stub | Stub |
| `POST /research/gaps` and `/validate` | Stub | Stub |

## 3. Why move away from LLM-only detection

Your methodology documents make three points that LLM-only detection works against:

- **"Sparse ≠ gap", and "observed scarcity ≠ actual research gap."** An LLM reading 20 abstracts can't tell the difference. It just writes plausible gap sentences.
- **Evidence-chain explainability.** Every gap should trace back to measurable signals: which papers, which sections, which counts. The current detector's main score is the LLM's self-reported `initial_confidence`, which makes up 30% of the rank.
- **RQ4: "which evidence signals distinguish genuine from false gaps?"** You can only answer that if detection produces separate, named signals.

Both papers find gaps structurally, not by asking an LLM:
- **ResearchRAG:** GapScore = 0.45·(1 − density) + 0.35·temporal staleness + 0.20·novelty, computed over concept clusters in a knowledge graph. The LLM is only used afterwards, to write up the clusters that score highest. Their simple edge-count heuristic reached Spearman ρ = 0.48 against expert rankings, and the GNN reached 0.71.
- **HySemRAG:** structured fields extracted per paper (methods, datasets, findings), plus LDA topic prevalence. Gaps come from methodology distributions and under-represented topics.

Your own documents also rule out building a GNN, Neo4j, LangGraph or multi-agent systems at this stage. So the plan below takes their signals and leaves out their infrastructure.

## 4. Proposed design: a signal-based candidate gap detector

Several independent, deterministic detectors each emit `CandidateGap` objects. Every candidate records **which signal produced it and the numbers behind it**. An LLM is optional and only used at the end, to turn a structured signal into a readable sentence. It never decides whether a gap exists. A template sentence is used when no LLM is available.

```
chunks (full-text sections, from main)  +  paper metadata (year, venue, authors)
        │
        ├─ D1 Explicit-statement miner       (Limitations / Future Work cue phrases)
        ├─ D2 Concept co-occurrence matrix   (method × dataset × task × population)
        ├─ D3 Temporal staleness             (publication-year trend per concept/cluster)
        ├─ D4 Topic density / novelty        (embedding clusters or LDA)
        └─ D5 Contradiction finder           (same setting, opposite result direction)
        │
        ▼
merge + de-duplicate  →  signal-weighted score  →  Top-K  →  EWRGV validation
        (optional: LLM or template only writes the gap_statement)
```

### D1. Explicit-statement miner (highest value, cheapest)

- **Input:** chunks whose section type is LIMITATIONS, FUTURE_WORK, DISCUSSION or CONCLUSION, from main's sectioner.
- **Method:** regex and cue-phrase patterns such as "remains unexplored", "has not been studied", "future work", "we did not evaluate", "limited to", "only tested on", "is left for future", and "lack of". Extract the noun phrase that follows each match. Then group similar statements across papers, using embedding cosine similarity or token overlap.
- **Signals:**
  - how many independent papers state it
  - which section types it came from
  - how recent the statements are
  - whether a later paper in the corpus mentions the same phrase in its Methods section (an early sign that someone has already addressed it)
- **Why first:** it's directly grounded, easy to explain ("3 papers' Future Work sections say X"), and matches the section-aware direction in your methodology document (section 15).

### D2. Concept co-occurrence and combination gaps (the ResearchRAG density idea, without the GNN)

- **Input:** entity tags per chunk (method, dataset, task, metric, population or language).
- **Tagging, cheapest first:**
  1. a dictionary built from query understanding's `methods`, `concepts`, `entities`, `datasets`, `metrics`, `terminology_variants` and `method_variants` (already produced, but not stored yet)
  2. SciSpaCy `en_core_sci_lg`, the model ResearchRAG uses
- **Method:** build co-occurrence matrices such as method × dataset and method × task. A **combination gap** is a cell with a count near 0 whose row and column are each well represented. For example, "transformers are used on datasets A, B and C, but never on D, although D appears in 12 papers". Score it with a density term (1 − observed/expected), where expected comes from the product of the row and column marginals.
- **Gap types covered:** Methodological, Dataset, Population, Evaluation (method × metric) and Geographic (if region or language entities are tagged).

### D3. Temporal staleness (ResearchRAG's τ)

- **Input:** publication years per concept or per cluster.
- **Method:** compare the latest year and the publication trend for a concept with the corpus as a whole. A concept with an old latest year, or a falling share of papers, scores high on staleness. Combined with D1 ("future work" stated years ago and no follow-up in the corpus), this is a strong signal.
- **Gap type:** Temporal.

### D4. Topic density and novelty (HySemRAG's LDA, ResearchRAG's density and η)

- **Input:** chunk embeddings from main's `/represent`.
- **Method:** cluster the embeddings (k-means or HDBSCAN), or run LDA on the chunk text. For each cluster compute density (paper count, cohesion), novelty (distance to the query centroid, or the share of recent papers) and staleness from D3. Score with ResearchRAG's form, α(1 − ρ) + βτ + γη, as a **heuristic baseline**. No trained GNN is needed, and their reported heuristic result (ρ = 0.48) gives you a published point to compare against.
- **Caveat ("sparse ≠ gap"):** a low-density cluster is only a *hypothesis*. That's exactly what EWRGV coverage analysis must test.

### D5. Contradiction finder

- **Input:** chunks that share the same (method, dataset or task) tags from D2.
- **Method:** a cue lexicon for result direction ("outperforms", "fails to", "no significant", "worse than"), or a small NLI cross-encoder such as `cross-encoder/nli-deberta-v3-base`, which classifies a pair of sentences and generates nothing. Opposite directions in the same setting give a Contradictory-evidence candidate.
- **Gap type:** Contradictory evidence.

### Merging and scoring

- Merge candidates from different detectors that point at the same concept set, by entity overlap or statement similarity. Agreement between detectors is itself a signal.
- Score per candidate (all values in [0, 1], weights set in config):
  - explicit-statement support (D1)
  - combination sparsity (D2)
  - staleness (D3)
  - cluster density and novelty (D4)
  - contradiction strength (D5)
  - independence (distinct papers, distinct venues and authors)
- The teammate's `CandidateEvidenceScorer` and `CandidateRanker` can be reused. Replace `initial_confidence` with this signal score, and keep the diversity and duplicate handling.
- Each `CandidateGap.metadata["signals"]` keeps the raw numbers. That gives you the evidence chain and the RQ4 feature table.

### Where the LLM detector fits

Keep the teammate's LLM `GapDetector` as **one baseline**, not the core:

| Configuration | Purpose |
|---|---|
| LLM-only detector (current branch) | Baseline 1 |
| Signal detectors only (D1–D5) | Proposed detection |
| Signal + LLM merged | Ablation |
| Each of the above, with and without EWRGV validation | RQ1: does validation reduce false positives? |

## 5. How this fits the existing code

| Need | Source |
|---|---|
| Full-text chunks with section types | Main: `ingestion/sectioning`, `ingestion/chunking` (`PaperChunk`) |
| Embeddings and the BM25 index | Main: `ingestion/representation`, `retrieval/bm25_index.py` |
| Paper years, venues, authors | Collection on both branches; the teammate's `CollectionCorpusStore` metadata |
| Concept dictionary | `query_understanding` output. It needs to be **stored** per run (current gap: `/query` results are discarded). |
| Candidate model, ranking, evidence scoring | Teammate branch: `CandidateGap`, `CandidateRanker`, `CandidateEvidenceScorer` |
| Interface | `domain/interfaces.py` `GapDetector` protocol. Each D-detector implements it, and a `CompositeGapDetector` merges their output. |

New code would live in `app/gap_detection/signals/` (one module per detector, plus `composite.py`), and in `app/ingestion/entities/` for tagging.

## 6. Plan, in order

1. **Merge the branches.** Rebase the teammate's branch onto main. Keep main's ingestion, embeddings and index, and the teammate's `gap_detection/` and pipeline stages. Add a `PaperChunk` → `DocumentChunk` adapter. Point `RetrievalService` at the real corpus instead of the stubs. Nothing below works until detection can see full-text sections.
2. **Store query understanding per run.** Save it, and link the `/query` ID to the `/collect` `run_id`. D2 and validation both need the variants.
3. **D1, the explicit-statement miner.** It gives the biggest early win and has a clear evidence trail.
4. **Entity tagging**, then **D2, the co-occurrence matrix**. Start with the dictionary and add SciSpaCy if recall is poor.
5. **D3 temporal staleness and D4 cluster density** (heuristic GapScore).
6. **D5 contradiction finder** (cue lexicon first, NLI model second).
7. **Composite detector and signal-weighted ranking.** Drop `initial_confidence` from the rank.
8. **Wire the APIs:**
   - `POST /research/gaps`: run detection for a `run_id`, store the candidates, return them with their signals.
   - `GET /research/gaps/{run_id}`: fetch the stored candidates.
   - `POST /research/validate`: start EWRGV for the Top-K, as a background job like `/acquire`.
   - `GET /research/validate/status/{job_id}`: validation job status.
   - Optional `POST /research/pipeline/run`: one call from question to ranked gaps.
9. **EWRGV validation**, the contribution: verification queries from the representation plus query-understanding variants; direct, terminology, method and counter searches **against Semantic Scholar and OpenAlex**, not only the local corpus; evidence classification; coverage; a confidence formula that replaces the 0.0; explanation.
10. **Evaluation set:** 30–50 gaps labelled genuine or spurious. Use it to compare the baselines and run the ablations above.

## 7. Decisions to make before building

- **Entity tagging:** query-understanding dictionary only (recommended to start), or SciSpaCy from the beginning.
- **Clustering for D4:** embedding k-means or HDBSCAN (recommended, since it reuses main's embeddings), or LDA like HySemRAG.
- **LLM's role in detection:** wording only (recommended), or keep it as one of several detectors that are merged.
