# Candidate Gap Identification — Phase 3 Documentation

## Status and limitation (read first)

| Component | Status |
|---|---|
| `GapDetector` (`app/gap_detection/detector.py`) | ✅ Implemented, unit-tested with a mocked LLM |
| `GapRepresentationBuilder` (`app/gap_detection/representation.py`) | ✅ Implemented |
| `CandidateRanker` (`app/gap_detection/ranking.py`) | ✅ Implemented (deterministic heuristic) |
| Pipeline Stage 3 — collection-backed corpus (`app/ingestion/corpus/collection_corpus.py`) | ✅ Abstract level only: one title + abstract chunk per unique collected paper (no PDF full text) |
| Pipeline Stages 4, 6, 7 (`app/orchestration/research_pipeline.py`) | ✅ Wired (Stage 4 uses BM25 over the collected corpus when a collection ran, else the injected retriever) |
| `POST /api/v1/research/gaps` | 🔲 Unchanged placeholder (see [API exposure](#api-exposure)) |

> ⚠️ **Collected papers are used at abstract level only.**
> When Stage 2 produces a `CollectionResult`, Stage 3 builds a
> `CollectionCorpusStore` (`app/ingestion/corpus/collection_corpus.py`): one
> title + abstract chunk per unique paper; papers without an abstract get no
> chunk. Duplicates are merged like the database deduplicator: records sharing
> any signature — normalised DOI, title + author words + year, or
> provider + provider ID — are merged transitively, keeping the longest
> abstract and filling missing DOI/year/authors from the other records.
> (Known shared limitation: author-name format or year differences without a
> common DOI/provider ID keep records apart.) Stage 4 retrieves from it with
> BM25, and the stub corpus is **not** used. If the collected corpus has no
> usable abstracts (empty or failed collection, or no abstracts), the run stops
> at `DOCUMENT_PROCESSING` with `PipelineResult.error` explaining why — including
> the provider-error count and the first provider error (credentials redacted,
> truncated to 200 characters). Corpus-construction, retriever-factory and
> unexpected retrieval failures end the run as `FAILED`, with the exception type
> and message in `error` and the traceback in the log.
> Without a collection (no `LiteratureCollectionService` injected), Stage 4 uses
> the injected retriever (the stub corpus in development) as before.
> `PipelineResult.corpus_source` / `corpus_stats` show which path ran.
>
> PDF full text (parsing → sectioning → chunking) is **not** implemented, so
> Limitations / Future Work sections — the strongest gap signals — are not yet
> seen. This path is tested offline with hand-written fictional abstracts only;
> it has not been run against live provider results or a real LLM.

No test in this phase calls a real LLM; all detector tests use a mocked
`LLMProvider`. Prompt quality against a real model has not been evaluated.

---

## Flow

```
ResearchQuery (+ QueryUnderstanding from Phase 2.1)
        ↓
Stage 4  retriever.retrieve(q) for the original query + each expanded query
        ↓   merge: best rank first, de-duplicate by chunk_id, cap at retrieval_top_k
        ↓   RetrievalResult → DocumentChunk (scores kept in chunk metadata)
DocumentChunk[]
        ↓
Stage 6  GapDetector.detect(query, chunks)
        ↓   ONE structured LLM call → GapDetectionLLMOutput (Pydantic-validated)
        ↓   grounding check on cited chunk IDs
CandidateGap[]  (unranked, with StructuredGapRepresentation)
        ↓
Stage 7  CandidateRanker.rank(candidates, top_k=TOP_K_GAPS)
        ↓
Top-K CandidateGap[]  → (Stage 8 EWRGV validation — not implemented)
```

Stage 5 (reranking) is skipped; the reranker is still a stub.

---

## GapDetector

**One LLM call** returns every candidate with:

| Field | Maps to |
|---|---|
| `gap_statement` | `CandidateGap.gap_statement` |
| `gap_type`, `secondary_gap_types` | `primary_gap_type`, `secondary_gap_types` (`GapType`) |
| `supporting_chunk_ids` | `relevant_passages` (+ `source_papers` derived from them) |
| `initial_confidence` | `initial_confidence` (clamped to [0, 1]) |
| `rationale` | `metadata["rationale"]` |
| `domain`, `population`, `methodology`, `outcome`, `temporal_scope`, `geographic_scope`, `key_concepts`, `synonyms` | `structured_representation` |

Files follow the `query_understanding/` pattern:
`prompts.py` (pure prompt builders) → `schemas.py` (LLM transport schema) →
detector maps to domain models.

**Behaviour**

- Empty or all-blank `chunks` → returns `[]` **without** calling the LLM.
- Input chunks are de-duplicated by `chunk_id`; only the first `max_chunks`
  (default 30) are sent, each truncated to `max_chunk_chars` (default 1500).
- **Grounding:** cited IDs not among the chunks sent to the LLM are removed and
  recorded in `metadata["invalid_chunk_ids"]`. A candidate with **no** valid
  cited ID is dropped as ungrounded.
- **Decorated IDs:** `resolve_chunk_id()` tries an exact match first, then strips
  harmless decoration copied from the prompt (`[chunk_id: c1]`, `` `c1` ``,
  `"c1"`, `chunk_id: c1`) and requires an exact, case-sensitive match against
  the supplied IDs. IDs that were not supplied are never accepted.
- **Lenient fields:** a bare string where a list is expected becomes a
  one-item list; other wrong types become empty lists / `None`.
- **Per-candidate validation:** a malformed candidate (not an object, blank
  statement) is dropped and logged (`dropped_candidates`) without discarding
  valid ones. If *every* candidate is malformed, or the top level is invalid
  (not an object, `candidate_gaps` not a list), `GapDetectionError` is raised.
- Gap-type labels are matched case-insensitively against `GapType` values/names
  plus a few aliases; unknown labels → `GapType.OTHER` with the raw label in
  `metadata["raw_gap_type"]`.
- At most `max_candidates` (typically `settings.MAX_CANDIDATE_GAPS`) are returned.
- Provider errors, non-dict responses, and an invalid top-level response raise
  `GapDetectionError`. The pipeline catches it and returns a `FAILED`
  `PipelineResult` that still contains the query and retrieved chunks.
- **Error details:** LLM calls go through `app/gap_detection/llm_call.py`
  (shared by the detector and `GapRepresentationBuilder`). The provider's cause
  (e.g. `Error code: 429 - rate limit`) is kept — truncated — in the error
  message (so it reaches `PipelineResult.error`), in full in `detail`, and in
  the warning log. Secrets are redacted first: `sk-…`/`xai-…` keys, bearer
  tokens, `api_key=`/`token:`/`password=`-style values, and the literal values
  of configured credential settings. Redaction is pattern-based, so an unusual
  secret format echoed by a provider could still slip through.

`initial_confidence` is the LLM's self-reported judgement of how strongly the
cited passages indicate the gap. It is **not** a validated confidence; the
post-validation confidence belongs to EWRGV.

## GapRepresentationBuilder

- `from_llm_output(statement, output)` — pure mapping used by the detector
  (no extra LLM call).
- `build(statement)` — its own LLM call for a standalone statement (e.g. a
  user-supplied gap). Raises `GapDetectionError` on a blank statement, missing
  provider, provider failure, or invalid output.

## CandidateRanker

Deterministic, no LLM. All parameters are constructor arguments.

```
base = w_conf     * initial_confidence
     + w_papers   * min(distinct_source_papers / paper_saturation,   1)
     + w_passages * min(distinct_passages      / passage_saturation, 1)
```

| Parameter | Default |
|---|---|
| `confidence_weight` / `paper_weight` / `passage_weight` (normalised) | 0.6 / 0.25 / 0.15 |
| `paper_saturation` / `passage_saturation` | 3 / 5 |
| `duplicate_threshold` (token Jaccard of statements) | 0.8 |
| `diversity_penalty` (per already-selected gap of the same primary type) | 0.1 |

1. Candidates are sorted by base score. A NaN `initial_confidence` scores as 0;
   ±inf is clamped.
2. **Near-duplicates** are dropped in favour of the higher-scored one; dropped
   IDs go to `metadata["ranking"]["merged_duplicates"]`. Two statements are
   duplicates when their letters/digits are identical (ignoring case,
   punctuation, and Unicode normalisation form), or when both have word tokens
   and their Jaccard similarity reaches the threshold (English stop words
   removed). Tokenisation is Unicode-aware (letters, combining marks, digits),
   so accented and non-Latin text (e.g. Tamil) is handled. Statements with no
   word tokens are never merged unless identical. **Limitation:** scripts
   written without spaces (e.g. Chinese) form one token per run, so only
   identical statements are detected as duplicates there.
3. **Diversity:** greedy selection on `base − penalty × same_type_count`.
4. Ties break on base score, then statement text, then `gap_id`.

Returned gaps are **copies** with `rank` (1 = highest) and
`metadata["ranking"]` (`base_score`, `diversity_penalty`, `adjusted_score`,
`distinct_papers`, `distinct_passages`, `merged_duplicates`). Inputs are not
mutated, so `PipelineResult.candidate_gaps` stays unranked.

**This heuristic does not measure novelty** or whether a gap exists in the wider
literature. It prioritises well-grounded, non-redundant, type-diverse
candidates for validation. Whether a gap is genuine is decided by EWRGV.

---

## Pipeline wiring

`ResearchPipeline` gained one optional argument, `retrieval_top_k=20`
(results per query, and cap on the merged chunk list). No factory in the app
constructs the pipeline yet; callers build it, e.g.:

```python
from app.core.config import settings
from app.gap_detection.detector import GapDetector
from app.gap_detection.ranking import CandidateRanker
from app.retrieval.service import RetrievalService      # stub corpus

pipeline = ResearchPipeline(
    query_understanding_service=qu_service,
    literature_collector=None,
    document_parser=None,
    retriever=RetrievalService(),           # retrieve(query, top_k) → RetrievalResult[]
    reranker=None,
    gap_detector=GapDetector(llm, max_candidates=settings.MAX_CANDIDATE_GAPS),
    candidate_ranker=CandidateRanker(),
    ewrgv_validator=None,
    top_k_gaps=settings.TOP_K_GAPS,
)
```

Any retriever whose `retrieve(query, top_k=...)` returns `DocumentChunk`s or
`RetrievalResult`s works; the pipeline normalises both.

---

## API exposure

`POST /api/v1/research/gaps` (`GapsRequest{job_id}` → `GapsResponse{job_id,
status, candidate_gaps: list[dict]}`) is intentionally unchanged. To serve real
results with that contract, the following is still needed:

1. **A pipeline factory / dependency** that builds `ResearchPipeline` from
   settings (LLM provider, retriever, detector, ranker), like
   `_get_qu_service()` does for Phase 2.1.
2. **Per-job result storage.** `job_id` must resolve to a stored
   `PipelineResult` (or its gaps). Today `JobStore` only tracks acquisition
   jobs, and `/query` does not run the pipeline or persist anything. A decision
   is needed on whether `/query` (or a new background job) runs Stages 1–7 and
   stores the result, and where (in-memory `JobStore` vs Supabase).
3. **A mapping** from `CandidateGap` to the `candidate_gaps` dicts (e.g.
   `gap.model_dump(mode="json")`), keeping domain models out of route code.
4. **Richer and persistent corpus data.** Stage 3 currently builds an in-memory,
   abstract-only corpus per run with run-local paper IDs (collectors assign
   random UUIDs, which differ across runs and from the database `paper_id`).
   Full-text (PDF) chunks and stable / database-backed paper IDs are needed
   before stored or cross-run results are meaningful.

---

## Tests

| File | Covers |
|---|---|
| `tests/unit/test_gap_detector.py` | valid output, invalid/ungrounded/decorated chunk IDs, empty input, provider/malformed errors, per-candidate validation, type mapping, caps, prompt content, stub-corpus chunks |
| `tests/unit/test_gap_detection_schemas.py` | schema coercion, per-candidate validation, strict top level, JSON schema shape |
| `tests/unit/test_gap_llm_call.py` | shared LLM call: cause preservation (e.g. 429), logging, secret redaction |
| `tests/unit/test_gap_representation.py` | `from_llm_output`, `build()` success and errors |
| `tests/unit/test_candidate_ranker.py` | scoring, near-duplicates (incl. Unicode/accents), NaN/inf confidence, diversity, top-k, determinism, config validation |
| `tests/unit/test_pipeline.py` (`TestPipelineGapStages`, `TestPipelineGapStagesCoverage`) | Stage 4 merge/conversion (both result types), Stage 6/7 wiring, failure handling incl. provider cause, stub-corpus runs with mocked LLM (incl. `retrieval_top_k=3`) |
