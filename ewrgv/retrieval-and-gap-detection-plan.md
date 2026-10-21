# Plan: retrieval fixes, then candidate gap detection

Status: **proposal, nothing implemented.** Builds on
[retrieval-pipeline-plan.md](retrieval-pipeline-plan.md) (the parts gap
detection needs; the one-call pipeline comes later) and on
[non-llm-gap-detection-plan.md](../analysis/non-llm-gap-detection-plan.md)
(detectors D1–D5).

## What exists today

- `app/gap_detection/` has an **LLM-only** detector: one LLM call over at most
  30 retrieved chunks. The LLM's self-reported confidence makes up most of the
  rank (weight 0.6 before normalisation in `CandidateRanker`).
- `CandidateEvidenceScorer` (relevance, section quality, recency, independence)
  and `CandidateRanker` (dedup and diversity) are deterministic and can be reused.
- `/gaps` and `/validate` are `not_implemented` placeholders.
- Retrieval works on run `e0b54bcc…`, but it still includes reference lists,
  can't filter by section, uses one query at a time, and re-downloads the
  corpus on every request.

---

## Part A: retrieval fixes (about 1 session)

**A1. Link `/query` to `/collect`**
- Migration 09 (shown to you before running, additive only) adds a `query_runs`
  table: the question, the understanding JSON (methods, datasets, metrics,
  entities, variants) and the expanded queries. `/query` returns its `query_id`.
- `/collect {query_id}` searches all stored expanded queries **plus the
  original question**. Passing `query` and `expanded_queries` by hand still works.
- `research_runs.query_id` links the run to its understanding. Gap detection
  uses it as its concept dictionary.
- `max_results_per_query` from the request is honoured (today `.env` wins), a
  `max_total_results` field is added, and the response says `capped: true`.
- A failed database save returns an error instead of a run_id that 404s later.

**A2. Clean the evidence pool**
- Exclude REFERENCES, ACKNOWLEDGMENTS and front-matter or page-header chunks
  (journal IDs, "Volume 8, Issue 2", licence text) from retrieval by default.
- Extend the heading guards for those page headers, and reclassify obvious
  cases to bring the 38% OTHER rate down.

**A3. Section-aware retrieval**
- `section_types` filter on `/retrieve/*`, with a `gap_evidence` preset
  (LIMITATIONS, FUTURE_WORK, DISCUSSION, CONCLUSION, RESULTS).
- `paper_ids` and `year_from` filters, which gap detection uses internally.

**A4. Faster retrieval**
- Cache each run's corpus in memory, refreshed when `/represent` finishes.

**A5. Multi-query retrieval**
- `retrieve_many(queries)`: the original question plus the expanded queries,
  fused with RRF, per-paper cap kept. This is the main input to gap detection.

---

## Part B: candidate gap detection (about 3–4 sessions, in order)

**B1. Run-level gap service and API**
- `POST /research/gaps {run_id, detectors?, top_k}` runs as a background job
  and stores candidates in a new `candidate_gaps` table (migration 10).
- `GET /research/gaps/status/{job_id}` and `GET /research/gaps/{run_id}`.
- Each candidate stores gap_type, statement, supporting chunk and paper IDs,
  and a `signals` dict with the raw numbers behind it (the evidence chain).

**B2. D1, explicit-statement miner** (first real detector, highest value)
- Scans LIMITATIONS, FUTURE_WORK, DISCUSSION and CONCLUSION chunks for cue
  phrases such as "remains unexplored", "has not been studied", "future work",
  "limited to", "only tested on" and "lack of".
- Extracts the stated gap, then groups similar statements across papers using
  the bge embeddings.
- Signals: number of independent papers, section types, recency, and whether a
  later paper's Methods section already addresses it.

**B3. Entity tagging + D2, combination gaps**
- Tag chunks with methods, datasets, tasks and metrics, using the stored query
  understanding as the dictionary (SciSpaCy only if recall turns out poor).
- Method × dataset and method × metric co-occurrence tables. A gap is a
  near-empty cell whose row and column are both well represented:
  density = 1 − observed/expected. This is ResearchRAG's density idea, without
  a graph database (the lightweight "knowledge graph").

**B4. D3 temporal staleness + D4 cluster density/novelty**
- Publication-year trends per concept.
- HDBSCAN on the chunk embeddings, scored with ResearchRAG's
  0.45·(1 − density) + 0.35·staleness + 0.20·novelty as the baseline.

**B5. Merge and rank**
- Merge candidates from different detectors that point at the same concepts;
  agreement between detectors is itself a signal.
- Weighted signals from config. **The LLM's self-confidence is dropped from
  the ranking.** `CandidateRanker` is reused for dedup and diversity.

**B6. LLM for wording only, plus a baseline**
- An LLM (or a template) writes the final gap sentence; it never decides that
  a gap exists.
- The current LLM detector stays as Baseline 1 for comparison: LLM-only vs
  signals vs both.

After B6: D5 (contradiction finder) and EWRGV validation (`/validate`).

**Tests:** unit tests for each detector on fixed fixture chunks, plus one
end-to-end run on `e0b54bcc` with a report of the candidates and their signals.

---

## Decisions for you

| # | Question | Options | Recommended |
|---|----------|---------|-------------|
| 1 | Entity tagging | query-understanding dictionary / SciSpaCy from the start | **dictionary first** |
| 2 | D4 clustering | HDBSCAN on existing embeddings / LDA (as in HySemRAG) | **HDBSCAN** |
| 3 | LLM role | wording only / also one merged detector | **wording only** |
| 4 | Scope of this approval | A + B1 + B2 then report back / all of Part B | **A + B1 + B2** |
