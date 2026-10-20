# Candidate Gap Identification — Phase 3 Documentation

## Status and limitation (read first)

| Component | Status |
|---|---|
| `GapDetector` (`app/gap_detection/detector.py`) | ✅ Implemented, unit-tested with a mocked LLM |
| `GapRepresentationBuilder` (`app/gap_detection/representation.py`) | ✅ Implemented |
| `CandidateRanker` (`app/gap_detection/ranking.py`) | ✅ Implemented (deterministic heuristic; blends in evidence scores when present) |
| `CandidateEvidenceScorer` (`app/gap_detection/evidence_scoring.py`) | ✅ Implemented, opt-in (Stage 6b): pre-validation evidence-weighted score — **not** validation (see [Evidence-weighted candidate scoring](#evidence-weighted-candidate-scoring)) |
| Pipeline Stage 3 — run corpus (`app/ingestion/corpus/run_corpus.py`) | ✅ Full-text PaperChunks from `/chunks` (`chunks.json`), with stored embeddings and the BM25 index from `/represent` |
| Pipeline Stages 4, 6, 7 (`app/orchestration/research_pipeline.py`) | ✅ Wired (Stage 4 retrieves from the run corpus; BM25 by default, hybrid with an injected factory) |
| `POST /api/v1/research/gaps` | ✅ Implemented as a separate signal-based API workflow; see [Knowledge Graph & Gap Detection API](gaps_kg_api.md) |

> **Scope:** This document also describes the injectable `ResearchPipeline`
> and its LLM-backed `GapDetector`. That legacy/general-purpose pipeline is
> distinct from the production `/research/gaps` job, which uses the KG signal
> detectors, rejects LLM options, and requires reranked retrieval evidence.

> **Corpus.** Gap detection runs on the run's `RunCorpus`: the section-aware,
> page-located `PaperChunk`s written by /acquire → /parse → /sections →
> /chunks, plus embeddings and the BM25 index from /represent.  Those phases
> are background jobs, so the corpus is passed in — `run(query, corpus=...)`,
> `run_on_corpus(query, corpus)`, or a `corpus_loader(run_id)` called after
> Stage 2 (e.g. wrapping `RunCorpusLoader.load`).
>
> * If a collection ran but no corpus is available, the run stops after
>   Stage 2 with `PipelineResult.error` listing the processing steps (and the
>   provider-error summary).  The injected retriever is **not** used then, so
>   gaps never come from papers outside the run.
> * An empty corpus (no chunked papers) stops at `DOCUMENT_PROCESSING`.
> * Corpus-loading, retriever-factory and unexpected retrieval failures end the
>   run as `FAILED`, with the exception type and message in `error` and the
>   traceback in the log.
> * Without a collection or corpus, Stage 4 uses the injected retriever.
>
> `PipelineResult.corpus_source` (`run_corpus` / `injected_retriever`) and
> `corpus_stats` show which path ran.  Because chunks carry `section_type`,
> Limitations / Future Work / Discussion / Conclusion passages — the strongest
> gap signals — are labelled as such in the detection prompt and in evidence
> scoring.  This path is tested offline with hand-written fictional chunks; it
> has not been run against a real LLM.

No test in this phase calls a real LLM; all detector tests use a mocked
`LLMProvider`. Prompt quality against a real model has not been evaluated.

---

## Flow

```
ResearchQuery (+ QueryUnderstanding from Phase 2.1)
        ↓
Stage 4  retriever.retrieve(q) for the original query + each expanded query
        ↓   merge: best rank first, de-duplicate by chunk_id, cap at retrieval_top_k
        ↓   each hit resolved to the corpus's PaperChunk (hits outside the corpus dropped);
        ↓   score / rank / method / query kept in PipelineResult.retrieval_provenance
PaperChunk[]
        ↓
Stage 6  GapDetector.detect(query, chunks)
        ↓   ONE structured LLM call → GapDetectionLLMOutput (Pydantic-validated)
        ↓   grounding check on cited chunk IDs
CandidateGap[]  (unranked, with StructuredGapRepresentation,
                 supporting + contradicting passage IDs)
        ↓
Stage 6b (optional) CandidateEvidenceScorer.score_all(candidates, chunks)
        ↓   evidence_assessment: score, status, features, references, explanation
        ↓
Stage 7  CandidateRanker.rank(candidates, top_k=TOP_K_GAPS)
        ↓
Top-K CandidateGap[]  → (Stage 8 EWRGV validation — not implemented)
```

Stage 5 (reranking) is skipped in this particular `ResearchPipeline` flow; the
separate API retrieval service has a CrossEncoder reranker. The `/research/gaps`
API uses that reranker for bounded query/expansion context before its signal
detectors run. See [retrieval API docs](retrieval_api.md).

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

When a candidate carries an `evidence_assessment`, its base score is blended
with the evidence score: `base = (1 − evidence_weight) · heuristic +
evidence_weight · evidence_score` (`evidence_weight` default 0.5; 0 ignores
assessments). Candidates without an assessment are scored exactly as before.
The heuristic's paper/passage breadth and the evidence score's independence
feature both reward multi-paper support, so that signal is partly counted twice
when both are used.

---

## Evidence-weighted candidate scoring

`CandidateEvidenceScorer` (`app/gap_detection/evidence_scoring.py`) assigns each
candidate a **pre-validation** `CandidateEvidenceAssessment`
(`app/domain/models/candidate_evidence.py`). It is a prioritisation signal
computed from evidence the pipeline already retrieved. It does not search for
new evidence, does not establish novelty, and never sets a `GapClassification`;
`is_validated` is always `False`. EWRGV validation (`app/validation/`) is
unchanged and remains responsible for deciding whether a gap is genuine.

### Evidence association

* **Supporting** passages = the candidate's `relevant_passages` (cited by the
  detector as motivating the gap, e.g. a stated limitation).
* **Contradicting** passages = the new `contradicting_passages`: the detector
  now also asks the LLM for `contradicting_chunk_ids` — retrieved passages
  suggesting the gap may already be addressed. They are grounded like
  supporting IDs (only supplied chunk IDs are accepted; others are recorded in
  `metadata["invalid_contradicting_chunk_ids"]`).
* A passage cited as **both** is kept only as counter evidence
  (`metadata["ambiguous_chunk_ids"]`); if that leaves no support, the candidate
  is dropped as ungrounded.
* Labels come from the detector LLM; the scorer does not judge stance itself.

### Features (only from available data)

| Feature | Definition | Data used |
|---|---|---|
| relevance | per passage: share of gap terms (statement + key concepts, Unicode word tokens, English stop words removed) found in the passage; candidate = mean | passage text |
| quality | per **supporting** passage only: 1.0 if the chunk's `section_type` is LIMITATIONS, FUTURE_WORK, DISCUSSION or CONCLUSION; for UNKNOWN/OTHER types, 1.0 if the section heading contains any of `limitation`, `future work`, `future direction`, `discussion`, `conclusion`, `open problem`, `open challenge`, `threats to validity` (case-insensitive substring); else 0.5 (values and heading list configurable); candidate = mean. Counter-evidence passages get no quality value | `PaperChunk.section_type`, else heading |
| recency | per passage: `max(0, 1 − max(0, reference_year − year) / 10)` — 1.0 for a paper from the reference year, decreasing by 0.1 per year older, 0 at 10+ years; a paper **newer** than the reference year (possible only with an explicit `reference_year`) is capped at 1.0. Candidate = mean over passages with a year; **unavailable** if none have a year | the paper's `publication_year` (via the corpus paper lookup) |
| independence | `min(distinct supporting papers / 3, 1)` | paper IDs (after corpus de-duplication) |

**Reference year.** If `reference_year` is configured (a positive integer; 0,
negative, non-integer or boolean values are rejected) it is used. Otherwise one
reference year is resolved per scoring call: the latest valid publication year
across **all** chunks supplied to `score_all()` (via the paper lookup), shared by every candidate in that call so their recency — and scores —
are comparable. A standalone `score()` call resolves it the same way from all of
its `chunks_by_id`. Passages without a year still get no recency (the feature is
excluded, not set to 0). Results never depend on the wall clock.

Citation counts are **not** scored: they
are a popularity proxy confounded with paper age, and are missing for many papers. When a provider supplied one, it is shown on
the evidence item for information only.

### Formula

```
strength_i    = weighted mean of passage i's available features
                (supporting: relevance, quality, recency; counter: relevance, recency)
support_score = Σ w_f · feature_f / Σ w_f      over AVAILABLE candidate features
ratio         = Σ strength_counter / (Σ strength_support + Σ strength_counter)
                (0 when both sums are 0)
score         = support_score × (1 − contradiction_penalty × ratio)
```

Default weights: relevance 0.35, quality 0.20, recency 0.15, independence 0.30
(any non-negative subset; renormalised over available features — the
assessment's `weights` shows the effective values). The **relevance weight must
be positive**: relevance is the only passage feature available for every
supporting and counter passage, so without it passage strengths — and the
contradiction ratio — could be 0 or meaningless (e.g. `{"independence": 1.0}`
or `{"quality": 1.0}` raise `ValueError`). `contradiction_penalty` defaults to
1.0. All parameters are constructor arguments and validated.

**Conservative bias toward contradiction.** Supporting passages include the
section-quality feature (0.5 outside explicit-gap sections), counter passages do not, so the
same passage carries more strength as counter evidence than as support (e.g. an
non-limitation passage with full term overlap and the newest year: 1.0 as counter vs 0.857 as
support with default weights). This deliberately errs toward discounting a
candidate gap rather than over-claiming it.

### Missing and contradictory evidence

| Situation | Status | Score |
|---|---|---|
| ≥1 supporting passage, ratio = 0 (no counter evidence, or only zero-strength counter passages) | `supported` | `support_score` |
| ratio > 0 and < 0.5 | `contested` | reduced by the ratio |
| ratio ≥ 0.5 | `contradicted` | reduced by the ratio |
| no available supporting passage | `insufficient_evidence` | **0.0** |

A counter passage has zero strength when it shares no gap terms and has no
positive recency; it does not change the score or status, and the explanation
lists it as having had no effect.

* Missing evidence is never treated as evidence that the gap exists: no support
  → score 0, and the explanation says so.
* The absence of counter evidence does not raise the score (it only means none
  was retrieved; the explanation says this does not confirm the gap).
* Cited passages that are not among the available chunks are listed in
  `missing_chunk_ids` and ignored.

### Output

Each assessment contains `score`, `status`, `support_score`,
`contradiction_ratio`, per-feature values and effective weights, the supporting
and contradicting evidence items (chunk ID, paper ID, section, stance,
relevance, quality, recency, strength, year, title, DOI, informational citation
count), `missing_chunk_ids`, and an `explanation`, e.g.:

> Evidence score 0.47 (contested): 2 supporting passage(s) from 2 independent
> paper(s); support score 0.68 from relevance 0.64 (weight 0.35), quality 0.75
> (weight 0.20), recency 0.70 (weight 0.15), independence 0.67 (weight 0.30).
> 1 contradicting passage(s) carry 31% of the evidence strength and reduce the
> score by 0.21. Candidate-level evidence score from retrieved passages only;
> not a validation of the gap.

### Enabling it

`ResearchPipeline(..., evidence_scorer=CandidateEvidenceScorer())`. With
`evidence_scorer=None` (default) the pipeline behaves exactly as before. Scorer
failures end the run as `FAILED` (`"Candidate evidence scoring failed: …"`),
keeping the unscored candidates in `candidate_gaps`.

### Limitations

* **Lexical relevance**: token overlap, not semantic similarity; paraphrases
  score low.
* **Stance comes from the LLM**: supporting/contradicting labels are only as
  good as the detector's judgement, and only retrieved passages can be labelled.
  No dedicated counter-evidence search is run (that belongs to EWRGV
  validation).
* **Section typing is heuristic**: quality depends on the section classifier
  (`app/ingestion/sectioning/classifier.py`); a mislabelled heading changes the
  quality value.  Independence is capped by the number of retrieved papers.
* **Recency is a heuristic**: a recent limitation statement is treated as more
  likely still open; it is not proof. Recency is relative to the newest paper in
  the scoring call, so scores from different calls (different retrieved sets)
  are not directly comparable unless `reference_year` is fixed.
* **Counter evidence weighs more than support** (no section-quality feature for
  counter passages) — a deliberate conservative bias, see above.
* **Weights are design defaults**, not calibrated against labelled data.
* **Not tested with a real LLM** — all tests are offline with fixed payloads.

---

## Pipeline wiring

Relevant `ResearchPipeline` arguments: `retrieval_top_k=20` (results per
query, and cap on the merged chunk list), `retriever_factory(corpus)`
(default: BM25 via `RetrievalService(corpus)`), `corpus_loader(run_id)` and
`evidence_scorer`.  No factory in the app constructs the pipeline yet;
callers build it, e.g.:

```python
from app.core.config import settings
from app.gap_detection.detector import GapDetector
from app.gap_detection.ranking import CandidateRanker
from app.ingestion.corpus.run_corpus import RunCorpusLoader
from app.retrieval.service import RetrievalService

corpus = await RunCorpusLoader(paper_repo, storage_client, embedding_repo).load(run_id)

pipeline = ResearchPipeline(
    query_understanding_service=qu_service,
    literature_collector=None,
    document_parser=None,
    retriever=None,
    reranker=None,
    gap_detector=GapDetector(llm, max_candidates=settings.MAX_CANDIDATE_GAPS),
    candidate_ranker=CandidateRanker(),
    ewrgv_validator=None,
    top_k_gaps=settings.TOP_K_GAPS,
    retriever_factory=lambda c: RetrievalService(c, embedding_provider),  # hybrid
)
result = pipeline.run(question, corpus=corpus)
```

`RetrievalService.retrieve` is synchronous; inside an event loop, pre-embed
the expanded queries (or run the pipeline in a worker thread) so the
embedding provider is not called from the loop.  Any retriever whose
`retrieve(query, top_k=...)` returns `PaperChunk`s or `RetrievalResult`s
works; with a corpus, both are resolved to the corpus's PaperChunks.

---

## API exposure

`POST /api/v1/research/gaps` is an implemented KG signal workflow and is not
the `ResearchPipeline` described above. It persists results through the KG
repository and reports job state through `/gaps/status/{job_id}`. Its request
and failure contract, including fixed-false LLM flags and reranker requirement,
is documented in [gaps_kg_api.md](gaps_kg_api.md). The process-local job-state
limitation and chunk-ID stability considerations remain relevant operational
constraints.

---

## Tests

| File | Covers |
|---|---|
| `tests/unit/test_gap_detector.py` | valid output, invalid/ungrounded/decorated chunk IDs, empty input, provider/malformed errors, per-candidate validation, type mapping, caps, prompt content (incl. section type), run-corpus chunks |
| `tests/unit/test_gap_detection_schemas.py` | schema coercion, per-candidate validation, strict top level, JSON schema shape |
| `tests/unit/test_gap_llm_call.py` | shared LLM call: cause preservation (e.g. 429), logging, secret redaction |
| `tests/unit/test_gap_representation.py` | `from_llm_output`, `build()` success and errors |
| `tests/unit/test_candidate_ranker.py` | scoring, near-duplicates (incl. Unicode/accents), NaN/inf confidence, diversity, top-k, determinism, config validation, evidence-score blending |
| `tests/unit/test_candidate_evidence_scoring.py` | evidence scoring: exact formula values, item features, weight configuration (incl. required relevance weight), missing evidence, contradictory and zero-strength counter evidence, recency renormalisation, shared/explicit reference year and its validation, corpus paper lookup, ranking with evidence |
| `tests/unit/test_pipeline_run_corpus.py` | collection → corpus_loader → detection, provenance, section types in the prompt, Stage 6b with corpus paper years, hybrid factory, missing/empty corpus, loader/factory/retrieval failures, hits outside the corpus |
| `tests/unit/test_run_corpus.py` | RunCorpus store, embeddings, lazy BM25; RunCorpusLoader with mocked repos: chunks.json download, skipped papers, embeddings, persisted / stale / unreadable BM25 index |
| `tests/unit/test_pipeline.py` (`TestPipelineGapStages`, `TestPipelineGapStagesCoverage`) | Stage 4 merge/conversion (both result types), Stage 6/7 wiring, failure handling incl. provider cause, sample-corpus runs with mocked LLM (incl. `retrieval_top_k=3`) |
