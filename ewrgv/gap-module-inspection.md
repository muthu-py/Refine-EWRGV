# Gap detection module: inspection after integration

Code: `ewrgv/app/gap_detection/` as merged on `integrate/gap-detection`
(Pavithran0630's commits `1ba825d`, `f2cd629`, adapted to PaperChunk in
`7005b45`). The earlier, pre-merge review is in
[gap-detection-and-validation-status.md](gap-detection-and-validation-status.md);
this note covers what is true **now**, on top of the full-text corpus and
today's retrieval fixes.

## What it is

An **LLM-proposes, code-checks** design:

| Step | File | What happens |
|---|---|---|
| Retrieve | `orchestration/research_pipeline.py` `_retrieve_chunks` | original + expanded queries, each top-20; merged by best rank, capped at **20 chunks** |
| Detect | `detector.py` + `prompts.py` + `schemas.py` | **one LLM call** over ≤30 chunks (≤1500 chars each). Returns statement, gap type (8-type taxonomy), supporting/contradicting chunk IDs, `initial_confidence`, structured representation |
| Ground | `detector.py` | cited IDs must be ones it was sent; candidates with no valid ID are dropped; an ID cited both ways counts as contradicting only |
| Score (optional) | `evidence_scoring.py` | relevance 0.35, section quality 0.20, recency 0.15, independence 0.30, then a contradiction penalty; status SUPPORTED / CONTESTED / CONTRADICTED / INSUFFICIENT_EVIDENCE |
| Rank | `ranking.py` | 0.6·LLM confidence + 0.25·paper breadth + 0.15·passage breadth, blended 50/50 with the evidence score; dedup by token Jaccard ≥ 0.8; type-diversity penalty; Top-5 |

Tests: 132 unit tests across detector, schemas, LLM call, representation and
evidence scoring, all with a fake LLM. **It has never been run against a real
LLM or a real corpus.**

## What's good

- The line between "candidate" and "validated" is explicit everywhere.
- Grounding is enforced in code, not trusted to the model.
- Per-candidate lenient parsing: one bad candidate doesn't sink the call.
- Secret redaction on provider errors, deterministic tie-breaks, thorough edge-case tests.

## What changed since the first review

| Earlier issue | Now |
|---|---|
| Abstract-only corpus, so section quality was constant | **Fixed by integration.** It runs on full-text PaperChunks with section types. But 38% of chunks are still OTHER, so the feature only partly works |
| `DocumentChunk` vs `PaperChunk` mismatch | **Fixed** (`7005b45`) |
| Reference lists as evidence | **Fixed today** in `/retrieve/*` (noise filter). But the pipeline's own retriever doesn't use those filters yet (see 4) |

## Problems that remain (by impact)

1. **The LLM decides that a gap exists and largely how highly it ranks.**
   The LLM's own confidence is 60% of the heuristic, so about 30% of the
   final score. That number is uncalibrated self-grading. For a thesis
   claiming evidence-weighted beats LLM judgement, this is the core weakness.
2. **Relevance is circular.** It measures token overlap between the gap
   statement and the passages the LLM wrote it from, so it is high by
   construction.
3. **Contradiction is opt-in.** Counter-evidence is only what the LLM chose
   to flag from the same 20 chunks. Nothing searches for disconfirming
   evidence, so `SUPPORTED` mostly means "the model didn't flag anything".
4. **The pipeline doesn't use today's retrieval fixes.** `_retrieve_chunks`
   calls `retriever.retrieve(q, top_k=20)` per query with plain BM25 by
   default. That means no `gap_evidence` section filter, no noise filter, no
   per-paper cap and no dense retrieval. The merge keeps the best rank per
   chunk, so with 9 queries the 20 slots are mostly rank-1 and rank-2 hits,
   and one paper can still dominate.
5. **Thin input.** 20 chunks from a 943-chunk run is about 2% of the
   corpus. Gaps stated in papers outside the top 20 are invisible.
6. **Not reachable.** `/gaps` and `/validate` are placeholders. The pipeline
   is synchronous (LLM call inside), so it can't be called from an async
   route as is.
7. **Independence counts papers**, not independent groups or datasets.
7b. **Section quality and relevance are narrow.** RESULTS doesn't count as an
   explicit-evidence section, and relevance uses word overlap even though
   bge embeddings are available.
7c. **No persistence.** Candidates aren't saved or linked to a run_id. Query
   understanding is only used if the pipeline re-runs `/query`, though it can
   now load the stored one from `query_runs`.
8. **Prompt fencing.** Passage text goes in unescaped, so a passage can forge
   a `[chunk_id: …]` header. Low risk because of grounding.
9. **Code health.**
   - `evidence_scoring.py` imports private helpers from `ranking.py`.
   - The stop-word list has only 30 words.
   - `ResearchPipeline` still takes the unused `literature_collector` and `document_parser`.

## How it fits the non-LLM plan

- **Keep:** the domain models (`CandidateGap`, `StructuredGapRepresentation`,
  `CandidateEvidenceAssessment`), grounding, `CandidateRanker`'s dedup and
  diversity, and the recency, section-quality and independence features.
- **Keep the LLM detector as Baseline 1** for the LLM-only comparison.
- **Change:**
  - drop `initial_confidence` from ranking, or ablate it;
  - feed detection from `aretrieve_many` with `gap_evidence` and the noise filter;
  - add the deterministic detectors D1 to D4 as the main path.
- **Then:** `/gaps` as a background job, and validation against external search.

## Suggested next check

Run the existing LLM detector once on run `e0b54bcc` with `gap_evidence`
retrieval to see real output: how many candidates come back, how many get
dropped as ungrounded, and what `initial_confidence` looks like. This costs
one LLM call and changes no code, and it gives a baseline before any rework.
