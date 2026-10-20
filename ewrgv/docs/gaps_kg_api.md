# Knowledge Graph & Gap Detection API

Phases P1–P4 of [`methodology/HYBRID_KG_GAP_METHODOLOGY.md`](methodology/HYBRID_KG_GAP_METHODOLOGY.md):
a per-run **academic knowledge graph**, multi-signal candidate gap detection, and evidence-weighted validation.

- **Prerequisites.** The run has been processed through
  `/query → /collect → /acquire → /parse → /sections → /chunks → /represent`
  (see [api_test_guide.md](api_test_guide.md)).
- **Required order.** Continue with `/kg/build → /gaps → /validate`. The API rejects
  `/kg/build` when the run has no chunked papers or no represented chunks (`409`),
  rejects `/gaps` when no graph exists (`409`), and rejects `/validate` when no
  candidate gaps have been saved (`409`). The Pipeline UI displays this full
  sequence and disables later stages until prerequisites complete.
- Candidate gaps become validated, uncertain, or unsupported only after the Phase 4 evidence and coverage checks.
- Signal detection remains the default. With explicit opt-in, `POST /gaps` can also add grounded LLM-proposed candidates and/or use an LLM to refine signal-candidate wording. An LLM proposal is marked as such and remains a hypothesis until validation.
- LLM self-reported confidence is displayed as proposal metadata only; it is never treated as evidence or used as the signal candidate score. Proposals must cite retrieved corpus chunks.
- Reranking is required for this endpoint. If the CrossEncoder is disabled or unavailable, the job fails with an explicit error rather than grounding candidates against unreranked results.

All paths are under `/api/v1/research`.

| Method | Path | What it does |
|---|---|---|
| POST | `/kg/build` | Starts a background job that builds and stores the run's knowledge graph |
| GET | `/kg/status/{job_id}` | Status of a KG build job |
| GET | `/kg/{run_id}` | The stored KG; `view=summary\|entities\|statements\|claims\|communities\|papers\|cytoscape\|graphml` |
| POST | `/gaps` | Starts a background job: KG → detectors → GapScore → stored candidates |
| GET | `/gaps/status/{job_id}` | Status of a gap-detection job |
| GET | `/gaps/{run_id}` | Stored candidates; `view=summary\|full\|compare` |
| POST | `/validate` | Starts Phase 4 validation for the top 10 candidate gaps or selected `gap_ids` |
| GET | `/validate/status/{job_id}` | Status of a validation job |
| GET | `/validate/{run_id}` | Stored classifications, evidence, coverage and explanation chains |

**Storage.** Each run's KG and gap results are stored in `kg_graphs` and `gap_detection_results` (migration `10_kg_and_gaps.sql`). Phase 4 results use columns added by `11_gap_validation.sql`; external literature responses are cached in `external_search_cache` for seven days. Rebuilding/re-running replaces the latest result for that run.

**Phase 3 detectors.** `/gaps` defaults to D1 explicit statements, D2 combination sparsity, D3 temporal staleness, and D4 topic density/novelty. The topic detector uses scikit-learn HDBSCAN and falls back to KG concept communities when chunk clusters are degenerate. D3 combines run-corpus trends with OpenAlex trends when available. Use the `detectors` field to select a subset.

**Phase 4 sources.** Validation defaults to the run corpus, OpenAlex and Semantic Scholar. The APIs are free to query; Semantic Scholar works without a key at lower rate limits, while `SEMANTIC_SCHOLAR_API_KEY` is optional. Unavailable external sources are listed in the result and do not abort corpus-based validation. Search responses are cached.

### `POST /validate`

```json
{
  "run_id": "e0b54bcc-ebcd-4e77-ae6e-d8e4fb4ef90d",
  "gap_ids": null,
  "sources": ["corpus", "openalex", "semantic_scholar"],
  "per_query": 10,
  "max_iterations": 2
}
```

`gap_ids` omitted or null validates the top 10 candidates saved for the run; provide selected candidate IDs to validate only those. `sources` accepts any non-empty combination of `corpus`, `openalex`, and `semantic_scholar`. `max_iterations` defaults to the configured application limit. The job status summary reports class counts and unavailable sources. Fetch the complete persisted output after completion with `GET /validate/{run_id}`.

Before first use after updating the code, apply pending migrations with `python scripts/apply_migrations.py` and restart the API server.

Result claims are extracted deterministically from `RESULTS`, `DISCUSSION`, and `ABSTRACT` chunks. A claim requires a tagged method, a dataset or task, and a directional result cue; direction is `POSITIVE`, `NEGATIVE`, or `MIXED`. A metric and numeric value are included when detected. View them with `GET /kg/{run_id}?view=claims`; claims are also included as `ResultClaim` nodes in Cytoscape/GraphML exports. This is rule-based extraction and may miss wording outside its cue patterns.

**Jobs** are kept in memory. A restarted server forgets job ids, but the stored results stay readable through the GET endpoints.

---

## 1. Build the knowledge graph

### `POST /kg/build`

```json
{
  "run_id": "e0b54bcc-ebcd-4e77-ae6e-d8e4fb4ef90d",
  "query_id": "57393ebc-7f52-4b7d-9c3e-7ae1f6de28cd",
  "relevance_threshold": null
}
```

| Field | Required | Default | Meaning |
|---|---|---|---|
| `run_id` | yes | — | A run with chunks (and embeddings from `/represent`) |
| `query_id` | no | the run's linked query (`research_runs.query_id`) | A stored `/query` result. Its methods, datasets, metrics, entities and variants extend the entity vocabulary, and the question and expansions drive the relevance gate |
| `relevance_threshold` | no | run-relative | A fixed off-topic cut-off between 0 and 1. The default is `max(0.55, median similarity − 0.08)`; `0` disables the gate |

**Response `202`:**

```json
{
  "job_id": "4ae96edb-2eed-4804-96a1-2764b1a081ac",
  "run_id": "e0b54bcc-ebcd-4e77-ae6e-d8e4fb4ef90d",
  "status": "PENDING",
  "message": "Knowledge-graph build started. Poll GET /api/v1/research/kg/status/4ae96edb-..."
}
```

**Errors:** `404` unknown run, `409` no chunked or represented corpus data, `422` invalid body.

### What the build does

1. **Relevance gate.** Computes the cosine similarity between the question-plus-expansions centroid and each paper's title and abstract. Papers below the threshold are kept but marked `on_topic: false`, and are skipped from then on.
2. **Noise exclusion.** Reference lists, acknowledgments and publisher boilerplate are not tagged. This is the same filter retrieval uses.
3. **Entity tagging.**
   - **Types:** `METHOD, TASK, METRIC, MODALITY, DISEASE, DATASET, POPULATION, CONCEPT`.
   - **Sources:** seed lexicons (`app/knowledge_graph/lexicon.py`), the stored query understanding, and a regex for "<Name> dataset".
   - **Matching:** whole-word and longest match first. Short acronyms (DR, FA, SVM) match case-sensitively.
4. **Synonym merging.** Names of the same type with bge cosine ≥ 0.88, or the same normalised form, are merged. Curated lexicon entities never merge with each other. All surface forms are kept as `aliases`.
5. **Gap statements.** Sentences carrying a limitation, future-work or open-problem cue, from sections where authors discuss limitations. Contribution claims ("the proposed…", "to overcome X we…"), motivation sentences, table rows and page headers are dropped.
   - Strength = cue strength × section weight. Cue strengths: OPEN_PROBLEM 1.0, LIMITATION 0.8, FUTURE_WORK 0.6. Section weights: LIMITATIONS / FUTURE_WORK 1.0, CONCLUSION / DISCUSSION 0.9, INTRODUCTION 0.6, …
6. **Graph.** Paper-level co-occurrence edges with PMI, then **Louvain communities** over edges with ≥ 2 papers and PMI > 0.
7. **Entity flags.** An entity in ≥ 80% of papers is flagged `generic` (e.g. "detection", "accuracy"). An entity that is part of the question itself is flagged `query_term`. Detectors ignore both.

### `GET /kg/status/{job_id}`

```json
{
  "job_id": "4ae96edb-...",
  "run_id": "e0b54bcc-...",
  "kind": "kg_build",
  "status": "COMPLETED",
  "progress": "done",
  "created_at": "2026-10-19T13:30:21+00:00",
  "started_at": "2026-10-19T13:30:21+00:00",
  "completed_at": "2026-10-19T13:31:47+00:00",
  "summary": { "...": "same as stats below" },
  "error": null
}
```

- `status` is one of `PENDING`, `RUNNING`, `COMPLETED` or `FAILED`.
- `progress` names the current step.
- `404` means the job id is unknown, or the server has restarted since.

---

## 2. View the knowledge graph

### `GET /kg/{run_id}?view=summary` (default)

Real output for run `e0b54bcc`, trimmed:

```json
{
  "run_id": "e0b54bcc-ebcd-4e77-ae6e-d8e4fb4ef90d",
  "query_id": "57393ebc-7f52-4b7d-9c3e-7ae1f6de28cd",
  "built_at": "2026-10-19T13:31:47+00:00",
  "stats": {
    "papers": 32, "on_topic_papers": 29, "off_topic_papers": 3,
    "chunks_total": 943, "chunks_tagged": 693, "chunks_noise_skipped": 113,
    "entities": 145,
    "entities_by_type": {"METHOD": 41, "DATASET": 37, "CONCEPT": 17, "DISEASE": 15, "TASK": 11, "METRIC": 11, "MODALITY": 8, "POPULATION": 5},
    "entities_merged_as_synonyms": 2, "mentions": 9497, "cooccurrence_edges": 6262, "communities": 4,
    "gap_statements": 135, "gap_statements_by_cue": {"LIMITATION": 95, "FUTURE_WORK": 26, "OPEN_PROBLEM": 14}
  },
  "off_topic_papers": [
    {"paper_id": "99adfe1c-...", "title": "AI-Based Crop Disease Severity Assessment with Explainable AI: ...", "relevance": 0.75},
    {"paper_id": "de91fb1d-...", "title": "A guide to machine learning for biologists", "relevance": 0.6374}
  ],
  "top_entities": [
    {"id": "MODALITY:fundus photography", "name": "fundus photography", "type": "MODALITY", "papers": 28,
     "mentions": 419, "community": 0, "generic": true, "query_term": false,
     "aliases": ["CFP", "color fundus", "fundus", "fundus image"]}
  ],
  "communities": [
    {"id": 0, "label": "accuracy, fundus photography, convolutional neural network", "size": 41},
    {"id": 1, "label": "prediction, segmentation, generalization", "size": 29}
  ],
  "strongest_cooccurrences": [{"a": "image analysis", "b": "robustness", "papers": 18, "pmi": 0.2465}],
  "views": ["summary", "entities", "statements", "communities", "papers", "cytoscape", "graphml"]
}
```

### Other views

| View | Query parameters | Returns |
|---|---|---|
| `entities` | `entity_type=METHOD` (any type, case-insensitive), `limit` | Entities sorted by paper count, with aliases, community and flags |
| `statements` | `limit` | The strongest gap statements: sentence, cue class, cue, strength, section, section_type, pages, year, paper title, entities |
| `communities` | `limit` (entities per community) | Louvain communities with member names |
| `papers` | — | Every paper with `relevance`, `on_topic`, year, DOI, plus the `relevance_threshold` used |
| `cytoscape` | — | `{"elements": {"nodes": [...], "edges": [...]}}`, ready for Cytoscape.js or Sigma. Nodes are Paper, Entity, GapStatement and Community; edges are MENTIONS, CO_OCCURS (≥ 2 papers), STATES, ABOUT and IN_COMMUNITY |
| `graphml` | — | A GraphML file download (`application/xml`) for Gephi, yEd or Cytoscape desktop |

**Example:**

```
GET /api/v1/research/kg/{run_id}?view=entities&entity_type=METHOD&limit=10
```

**Errors:** `404` means no KG has been built for the run yet.

---

## 3. Detect candidate gaps

### `POST /gaps`

```json
{
  "run_id": "e0b54bcc-ebcd-4e77-ae6e-d8e4fb4ef90d",
  "query_id": "57393ebc-7f52-4b7d-9c3e-7ae1f6de28cd",
  "detectors": ["explicit", "combination"],
  "llm_wording": false,
  "include_llm_candidates": false,
  "include_llm_baseline": false,
  "rebuild_kg": false
}
```

| Field | Default | Meaning |
|---|---|---|
| `detectors` | both | `explicit` (D1) and/or `combination` (D2) |
| `llm_wording` | `false` | If true, asks the configured LLM to refine eligible signal-based gap wording using its retrieved supporting evidence. The original wording and provenance are retained; grounding checks can reject a rewrite |
| `include_llm_candidates` | `false` | If true, adds grounded candidates proposed by the LLM detector. Each candidate records its source and cited corpus evidence. Near duplicates merge with signal candidates and retain both sources |
| `include_llm_baseline` | `false` | If true, also stores the legacy LLM-only baseline for comparison. When either LLM option is enabled, the detector output is reused within that job |
| LLM configuration | — | Enabling any LLM option requires a configured provider API key; otherwise the request returns `503` |
| `rebuild_kg` | `false` | Rebuilds the KG first. The run must already have chunked/represented corpus data and a stored KG unless this flag requests a rebuild |

**Response:** `202`, in the same shape as `/kg/build`. Poll `GET /gaps/status/{job_id}`. On this run the job takes about 11 seconds once the KG exists.

### Detectors

**D1 — explicit-statement consensus.**
1. Gap statements that name at least one specific concept are clustered by meaning (bge, average-linkage cosine ≥ 0.72).
2. A second pass merges clusters about the same concepts (concept Jaccard ≥ 0.5 and centroid cosine ≥ 0.60).
3. Each cluster becomes one candidate. Its representative statement is the central, strongly cued, well-formed sentence.
4. Its signals are: `stating_papers`, `support = Σ strongest strength per paper`, `explicit = 1 − exp(−support/2)`, `cue_mix`, `latest_statement_year`, and `addressed_later` (a newer paper whose methods or results mention all the core concepts; reported as `counter_hints`).

**D2 — combination sparsity with link prediction.** For entity pairs of complementary types (method × modality, disease, dataset, task, population, concept; modality × disease; disease × population) where each entity appears in ≥ 3 papers:

- `lift = observed / expected`, where `expected = n_A · n_B / N`.
- `sparsity = 1 − min(lift, 1)`.
- `link_prediction` = Jaccard of the two entities' co-occurrence neighbourhoods, i.e. whether the concepts *should* meet.
- Candidates have `lift ≤ 0.35`, `expected ≥ 1.5` and `link ≥ 0.25`.

D2 statements are templates ("X is rarely combined with Y: n papers study X…"). Papers that do combine X and Y are listed as `counter_hints`.

**Fusion.**
- A D2 pair contained in a D1 candidate's concepts merges into it. Both detectors are then listed in `detectors_agreeing`, and the agreement bonus applies.
- A stand-alone D2 candidate picks up any gap statements whose entities include both of its concepts.

### GapScore v1

    E  explicit      = 1 − exp(−support/2)
    S  sparsity      = (1 − lift) × (0.5 + 0.5·link)  ×  min(1, on_topic_papers / 50)
    I  independence  = min(stating_papers / 3, 1)
    A  agreement     = (detectors − 1) / 3

    gap_score = (0.25·E + 0.25·S + 0.15·I) / 0.65 + 0.10·A        (capped at 1)

- **Small-corpus factor.** `min(1, N/50)` discounts sparsity measured in a small corpus. Sparsity alone is not evidence of a gap; external coverage is checked in phase P4.
- **Ranking** is by `gap_score`, minus 0.05 per already-ranked candidate of the same gap type, capped at 0.10. The result is `rank_score`.
- **Grounding gate.** The job first performs one bounded hybrid retrieval across the query and its expansions, reranks the candidate passages with the CrossEncoder, and uses those passages as textual grounding context. A textual statement must have cosine ≥ 0.60 to a supporting sentence or evidence chunk. If it remains below threshold, the job re-retrieves up to 3 times (hybrid retrieval with the gap-evidence filter); if it is still ungrounded, `status` becomes `"uncertain"`. D2 template statements are grounded *structurally*: both concepts need evidence passages. Job stats include reranked context candidate count and reranking status.
- **Gap type.** Rules over the concepts and the statement: Population, Dataset, Evaluation, Contradictory Evidence, otherwise Methodological.

---

## 4. View candidate gaps

### `GET /gaps/{run_id}?view=summary&top=10` (default)

Real output for run `e0b54bcc`, top 1, trimmed:

```json
{
  "run_id": "e0b54bcc-ebcd-4e77-ae6e-d8e4fb4ef90d",
  "query_id": "57393ebc-7f52-4b7d-9c3e-7ae1f6de28cd",
  "params": {"detectors": ["explicit", "combination"], "llm_wording": false, "include_llm_baseline": false,
             "llm_wording": false, "include_llm_candidates": false,
             "include_llm_baseline": false,
             "kg_built_at": "2026-10-19T13:31:47+00:00"},
  "stats": {
    "candidates_per_detector": {"explicit_statements": 26, "combination_sparsity": 10},
    "candidates_after_fusion": 36, "merged_multi_detector": 0, "grounded": 36, "uncertain": 0,
    "by_gap_type": {"Methodological": 21, "Dataset": 11, "Evaluation": 4}, "multi_paper": 10
  },
  "top_gaps": [
    {
      "rank": 1,
      "gap_statement": "A considerable number of existing approaches are designed for binary classification or single-disease detection, which restricts their applicability in practical clinical screening environments where multiple ocular diseases must be identified simultaneously.",
      "gap_type": "Methodological",
      "status": "candidate",
      "gap_score": 0.5579,
      "rank_score": 0.5579,
      "score_breakdown": {
        "explicit":        {"value": 0.8504, "weight": 0.3846, "contribution": 0.3271},
        "sparsity":        {"value": 0.0,    "weight": 0.3846, "contribution": 0.0},
        "independence":    {"value": 1.0,    "weight": 0.2308, "contribution": 0.2308},
        "agreement_bonus": {"value": 0.0,    "weight": 0.1,    "contribution": 0.0}
      },
      "detectors_agreeing": ["explicit_statements"],
      "concepts": ["screening (TASK, 18 papers)"],
      "stating_papers": 8,
      "supporting_statements": [
        {"paper": "An Adaptive Attention-Driven Deep Learning Framework with Optimized Segmentation for Multi-Class Diabetic Eye Disease Diagnosis",
         "year": 2026, "section": "2.1 Research Gap and Novelty Analysis", "section_type": "DISCUSSION", "pages": "5",
         "cue": "LIMITATION: single-disease",
         "quote": "Furthermore, a considerable number of existing approaches are designed for binary classification or single-disease detection, ..."},
        {"paper": "Artificial Intelligence in Retinal Disease Detection: Technological Advances and Clinical Applications Research",
         "year": 2025, "section": "Conclusion", "section_type": "CONCLUSION", "pages": "5-6",
         "cue": "LIMITATION: single-disease",
         "quote": "First, there will be a transition from single-disease screening to a comprehensive correlation analysis of ocular complications ..."}
      ],
      "evidence_passages": [
        {"paper": "Semi-Supervised Learning for Retinal Disease Detection: A BIOMISA Study", "year": 2025,
         "section": "6. Problems of the BIOMISA Macula Dataset", "section_type": "DATASET", "pages": "6-7",
         "role": "gap_statement", "snippet": "Based on these limitations of the existing works, ..."}
      ],
      "counter_hints": [],
      "grounding": {"grounded": true, "max_similarity": 0.9981, "method": "semantic", "attempts": 0},
      "signals": {"explicit_statements": {"statements": 13, "stating_papers": 8, "support": 3.8, "explicit": 0.8504,
                  "cue_mix": {"LIMITATION": 9, "FUTURE_WORK": 4}, "latest_statement_year": 2026,
                  "cluster_cohesion": 0.6761, "addressed_later": 0}}
    }
  ]
}
```

### Card fields

| Field | Meaning |
|---|---|
| `gap_score` / `rank_score` | Score before and after the type-diversity penalty |
| `score_breakdown` | value × weight = contribution for E, S, I and the agreement bonus; the contributions sum to `gap_score` |
| `detectors_agreeing` | Which detectors produced the candidate |
| `stating_papers` | Independent papers that state the gap |
| `supporting_statements` | The quoted gap sentences, with paper, year, section, section_type, pages and cue |
| `evidence_passages` | Chunks behind the candidate. `role` is `gap_statement`, `mentions <concept>` (D2) or `re-retrieved (grounding)` |
| `counter_hints` | Early counter-evidence from the run corpus: papers that combine the concepts (D2), or newer papers that address them (D1). Not scored; for EWRGV validation |
| `grounding` | Grounding-gate result |
| `signals` | Raw detector numbers |

### Other views and filters

| Parameter | Effect |
|---|---|
| `view=full` | Complete stored candidates, including source (`signal`, `llm`, or `signal+llm`), wording provenance, `structured_representation`, and matching LLM proposal metadata/evidence, plus `llm_baseline` when requested |
| `view=compare` | Signal-based top-N side by side with a previously stored LLM baseline, if one exists. Request `include_llm_baseline: true` to create/update it |
| `top=N` | Number of cards, 1–200 |
| `detector=explicit_statements \| combination_sparsity` | Only candidates produced by that detector |

**Errors:** `404` means no results yet; run `POST /gaps` first.

---

## 5. Typical session

```bash
RUN=e0b54bcc-ebcd-4e77-ae6e-d8e4fb4ef90d
QID=57393ebc-7f52-4b7d-9c3e-7ae1f6de28cd
API=http://127.0.0.1:8000/api/v1/research

curl -s -X POST $API/kg/build -H "Content-Type: application/json" -d "{\"run_id\":\"$RUN\",\"query_id\":\"$QID\"}"
curl -s $API/kg/status/<job_id>                    # repeat until COMPLETED
curl -s "$API/kg/$RUN"                              # summary
curl -s "$API/kg/$RUN?view=statements&limit=20"
curl -s "$API/kg/$RUN?view=claims&limit=20"
curl -s "$API/kg/$RUN?view=graphml" -o kg.graphml   # open in Gephi / yEd

curl -s -X POST $API/gaps -H "Content-Type: application/json" -d "{\"run_id\":\"$RUN\",\"query_id\":\"$QID\"}"
curl -s $API/gaps/status/<job_id>                  # repeat until COMPLETED
curl -s "$API/gaps/$RUN?top=5"
curl -s "$API/gaps/$RUN?detector=combination_sparsity&top=10"

curl -s -X POST $API/validate -H "Content-Type: application/json" -d "{\"run_id\":\"$RUN\"}"
curl -s $API/validate/status/<job_id>                 # poll until COMPLETED
curl -s "$API/validate/$RUN"                         # classifications + evidence chains
```

Swagger UI at `/docs` lists every endpoint, under the tag **Knowledge graph & gaps**.
