# EWRGV Evaluation Metrics Report

**Prepared:** 2026-10-20  
**Scope:** Repository evidence and the supplied HySemRAG and ResearchRAG papers. No production code or configuration was changed.

## 1. Executive summary

EWRGV implements dense, sparse/BM25 and hybrid retrieval, cross-encoder reranking, signal-based candidate generation/ranking, and evidence-weighted validation. The inspected repository does not contain a labeled evaluation set, reference relevance judgments, reference gap labels, saved validation outputs, or a metrics evaluation harness. Therefore Precision@K, Recall@K, MRR, MAP, nDCG, gap quality, validation classification quality/calibration, and measured end-to-end timing are **Not Measured** for EWRGV. Algorithms and tests are not performance results.

One saved LLM candidate artifact records 20 retrieved chunks, 6 proposed candidates and 5 top-ranked candidates for one run. This is a run output, not an evaluation: no ground truth or repeatable benchmark protocol accompanies it. No EWRGV full pipeline run was attempted because it requires configured database/storage and embedding/search providers; this report does not inspect or disclose secrets.

HySemRAG reports development-time similarity and quality-control results (arXiv preprint, 2025). ResearchRAG reports extensive retrieval, synthesis, gap-rating and timing results (Research Square preprint, 2026). These published figures are transcribed as reported, not reproduced. Their definitions, datasets, protocols and sample sizes differ from EWRGV and from each other; they are not direct head-to-head evidence.

## 2. EWRGV implementation and evaluation inventory

### Implementation inspected

- Retrieval: `app/retrieval/{service.py,semantic.py,bm25.py,fusion.py,reranking.py}`. Dense/sparse/hybrid retrieval and a cross-encoder reranker exist. Reranking code returns elapsed milliseconds for a call, but there is no saved benchmark timing series. Ranker unit tests check behavior, not relevance quality.
- Candidate generation/ranking: `app/gap_detection/signals/`, `app/gap_detection/{detector.py,ranking.py,evidence_scoring.py}`, and `app/knowledge_graph/{builder.py,jobs.py}`. Signals and heuristic scores rank hypotheses; there are no reference labels against which to compute gap precision/recall/F1.
- Validation/scoring: `app/validation/`, `app/evidence/`, `app/coverage/`, and `app/scoring/{confidence.py,classification.py}`. Validation records evidence, coverage, confidence and a three-way classification. This is implemented behavior, not evaluated classification accuracy or calibration.
- API and persistence: `app/api/routes/{research.py,gaps_kg.py}`, migrations, and `docs/SYSTEM_TEST_GUIDE.md`. The guide describes a staged live workflow and expected fields; it does not report a completed run.
- Test inventory: `tests/unit/` and `tests/integration/`. Integration tests use fixtures/mocks for external systems; their existence does not establish live-service success.

### Result artifacts inspected

- `data/artifacts/bm25/*/bm25_index.pkl`: seven serialized BM25 indexes. These indexes contain no evaluation judgments or metric summaries.
- `data/artifacts/gaps/e0b54bcc-ebcd-4e77-ae6e-d8e4fb4ef90d_llm_gaps.json`: one saved LLM candidate artifact, with 20 retrieved chunks, 6 candidate gaps and 5 top-k gaps. Its `error` field is null. No relevance labels, evaluation set, or measured quality metrics accompany it. Do not interpret those counts as a validated EWRGV benchmark.
- No evaluation CSV/JSON/log/notebook/report with EWRGV quality scores or persisted validation run was found in the inspected repository paths.

### Executed checks

- `python -m pytest tests/unit/test_reranking.py tests/unit/test_candidate_ranker.py tests/unit/test_candidate_evidence_scoring.py -q` — **113 passed in 1.16 s**. This is a unit-test result only; it is not a retrieval/gap/validation score.
- `python -m pytest tests -q` was attempted but did not finish or produce a final summary in the available run; it was interrupted. A broader focused command including `test_gaps_kg_api.py` likewise stalled after partial progress and was interrupted. No full-suite pass is claimed.
- No live API, database, external search, embedding, or complete end-to-end pipeline evaluation was run. The documented system-test guide requires configured PostgreSQL/Supabase, storage, providers/credentials and a sequential real run.

## 3. Exact EWRGV metrics table

| Component | Metric | Exact value | Dataset / sample size | Evaluation status | Evidence source | Reproducible? |
|---|---|---:|---|---|---|---|
| Retrieval | Precision@K | Not Measured | No labeled queries/judgments found | Not Measured | Retrieval modules; README lists evaluation as future work | No, labels absent |
| Retrieval | Recall@K, before/after reranking | Not Measured | No labeled queries/judgments found | Not Measured | `app/retrieval/`; `scripts/verify_retrieval.py` prints ranked examples only | No, labels absent |
| Retrieval | MRR, MAP, nDCG@K | Not Measured | No labeled queries/judgments found | Not Measured | No metric evaluator or relevance file found | No, labels absent |
| Retrieval | Retrieval/reranking latency | Not Measured | No recorded timing run | Not Measured; reranker exposes per-call elapsed ms in code | `app/retrieval/reranking.py` | No recorded measurements |
| Candidate generation | Candidate count in saved LLM artifact | 6 candidates; 5 in top-k list | One saved run; artifact also stores 20 retrieved chunks | Documented in raw artifact, not quality-evaluated | `data/artifacts/gaps/e0b54bcc-ebcd-4e77-ae6e-d8e4fb4ef90d_llm_gaps.json` | Yes for artifact counts only; not a benchmark |
| Candidate gaps | Precision, recall, F1, ranking metrics | Not Measured | No documented ground-truth gap set | Not Measured | No ground-truth labels found; `README.md` calls benchmarking future work | No, labels absent |
| Validation | VALID_GAP / UNCERTAIN_GAP / UNSUPPORTED_GAP counts | Not Measured | No persisted validation output found | Not Measured | Validation code and API schema exist; no result set found | No recorded outputs |
| Validation | Accuracy, precision, recall, F1, confusion matrix | Not Measured | No reference labels | Not Measured | No labeled validation evaluation found | No, labels absent |
| Validation | Confidence distribution/calibration | Not Measured | No recorded validation output/reference labels | Not Measured | `app/scoring/confidence.py` implements scoring only | No, outputs/labels absent |
| Validation | Evidence-type counts, coverage, success/failure counts | Not Measured | No validation result set | Not Measured | `app/domain/models/validation.py`; validation endpoints | No recorded outputs |
| Validation | Latency, verification searches per candidate | Not Measured | No recorded validation run | Not Measured | `app/validation/jobs.py` records search call counts in successful job summary, but no saved execution was found | No recorded run |
| System | API success/failure rate; stage and end-to-end time | Not Measured | No observed live run/log | Not Measured | `docs/SYSTEM_TEST_GUIDE.md` specifies expected workflow only | No live run |
| Ingestion | Collected/deduplicated/downloaded/parsed papers and chunks | Not Measured | No live run summary found | Not Measured | API schemas/documentation provide count fields; no saved run summary found | No live run |
| Tests | Focused reranking/ranking/evidence-scoring test pass rate | 113/113 passed (100% for this command) | 113 unit tests | Measured; code tests only | Command in §9; pytest output | Yes, command recorded |

## 4. HySemRAG published metrics

**Publication identity:** Alejandro Godinez, “HySemRAG: A Hybrid Semantic Retrieval-Augmented Generation Framework for Automated Literature Synthesis and Methodological Gap Analysis,” arXiv:2508.05666, submitted 1 August 2025; arXiv DOI [10.48550/arXiv.2508.05666](https://doi.org/10.48550/arXiv.2508.05666). This is a preprint. The supplied local PDF matches the title and author. The primary arXiv record verifies title, author, date, abstract and DOI.

The paper labels its main experiment “controlled developer testing rather than independent evaluation.” Reported results below are published as stated, not independently reproduced. The paper does not provide query-level raw outputs in the supplied repository.

| Published metric (original name) | Reported value | Dataset / sample | Location in paper |
|---|---:|---|---|
| Structured field extraction semantic similarity | 0.655 ± 0.178 (n=165) | HySemRAG development observations | §4.1, PDF p. 44 |
| PDF text chunking semantic similarity | 0.485 ± 0.204 (n=263) | HySemRAG development observations | §4.1, PDF p. 44 |
| Difference / significance | 35.1% higher; Mann–Whitney U, p < 0.000001 | Comparison above | §4.1, PDF p. 44 |
| Observations / testing sessions | 643 observations / 60 sessions | Controlled development testing | §4.1, PDF p. 44 |
| Evidence-quality classification | High: 100 (24.5%); medium: 258 (63.2%); low: 50 (12.3%) | Paper does not state a consistent denominator alongside these counts; counts sum to 408 | §4.1, PDF p. 44 |
| Agentic QA single-pass success rate | 68.3%; mean 2.33 iterations/query | Testing session observations; denominator not specified here | §4.1, PDF p. 45 |
| Quality-controlled acceptance rate | 61.3% (394 verified observations) | 394 verified observations | §4.1, PDF p. 45 |
| Citation accuracy; semantic similarity in quality-controlled analysis | 99.0% ± 4.5%; 1.000 ± 0.000 | 394 verified observations | §4.1, PDF p. 45 |
| Citation field accuracy: DOI; in-text; full citation; Zotero key | 74.4%; 73.9%; 73.9%; 72.3% | Component analysis; denominator not stated in passage | §4.2, PDF p. 45 |
| Vector database integration success | 99.6% (242/243 attempts) | 243 attempts | §4.2, PDF p. 45 |
| Temporal performance trend / daily range | −1.9% per day; daily valid-response range 100.0% to 25.0% | Across evaluation period; duration not specified in passage | §4.3, PDF p. 45 |
| Session-level response quality | 27/60 sessions (45.0%) ≥80%; 20/60 (33.3%) <50% | 60 sessions | §4.3, PDF p. 45 |
| Parsing completeness/failures | 518/643 (80.6%) complete; 125/643 (19.4%) failed | 643 observations; paper attributes 65 to empty metadata and 55 to malformed structure (5 other failures not categorized there) | §4.3, PDF p. 45 |
| Gap-analysis application corpus | 34 empirical studies; 102 ML method instances | Geospatial epidemiology literature on ozone exposure and cardiovascular disease | §4.4, PDF p. 45 |

The 643-observation total and some component denominators are not reconciled in the prose (e.g., 100+258+50 = 408). These are retained as paper-reported values, not recalculated. Semantic similarity is not Precision/Recall or a standard retrieval metric as defined in this report. The paper reports application findings about methods used in the 34-study corpus, but no labeled gap-identification precision/recall/F1 or human gap-quality score in the inspected results section. Those specific quality metrics are **NR (Not Reported)** in the paper.

## 5. ResearchRAG published metrics

**Publication identity:** Aritrik Ghosh, “ResearchRAG: A Knowledge-Graph-Enhanced Hybrid Multi-Agent RAG Framework for Academic Literature Analysis and Adaptive Research Gap Discovery,” Research Square preprint, posted 16 June 2026, DOI [10.21203/rs.3.rs-10028443/v1](https://doi.org/10.21203/rs.3.rs-10028443/v1). The supplied PDF identifies the author, date, DOI, and CC BY 4.0 license. It is a preprint, not an independently reproduced result here. No official implementation/evaluation artifact was present in this repository or located in the sources reviewed.

| Published metric (original name) | Reported value | Dataset / sample | Location in paper |
|---|---:|---|---|
| Precision@5 / Precision@10 | 0.793 / 0.831 (P@10 95% CI 0.817–0.845) | TREC-COVID; 50 expert-annotated topics; five repeats, means across seeds | Table IV, PDF p. 6 |
| Recall@10 / Recall@20 | 0.709 / 0.812 | TREC-COVID | Table IV, PDF p. 6 |
| MRR / NDCG@10 | 0.774 (95% CI 0.761–0.787) / 0.748 (95% CI 0.734–0.762) | TREC-COVID | Table IV, PDF p. 6 |
| BERTScore F1 / ROUGE-1 / ROUGE-L / BLEU-4 | 0.871 (95% CI 0.863–0.879) / 0.478 / 0.431 / 0.211 | SCIDOCS; 500 paper clusters, mean 4.2 papers; two PhD annotators, κ=0.69 | Experimental Setup §IV.A, PDF p. 5; Table V, PDF p. 6 |
| Hallucination rate | 0.078 (95% CI 0.068–0.088); Standard RAG 0.187; reported relative reduction 58.3% | SCIDOCS synthesis; 11,700 generated sentences reported in error analysis | Table V, PDF p. 6; §VI.B, PDF p. 7 |
| Gap detection expert Likert ratings (Relevance / Specificity / Novelty / Composite) | 4.21 / 3.98 / 3.94 / 4.04 (out of 5) | 250 custom gap queries; two PhD annotators, κ=0.67 | Experimental Setup §IV.A, PDF p. 5; Table VI, PDF p. 7 |
| GNN-RGS rank correlation with expert gap rankings | Spearman ρ=0.71; heuristic baseline ρ=0.48; paired t(248)=9.3, p<0.001 | 1,200 manually annotated concept clusters for training; gap-query evaluation described above | Contributions, PDF p. 2; §III.E, PDF p. 4; §V.C, PDF pp. 6–7 |
| Gap-rating baseline comparisons (composite out of 5) | Standard RAG 2.64; Agentic RAG 3.32; GraphRAG 3.27; AutoSurvey 2.97; ResearchRAG 4.04 | Custom gap queries | Table VI, PDF p. 7 |
| Ablation: gap composite | Full 4.04; removing GNN-RGS 3.41; other configurations 2.38, 2.14, 2.79, 3.89, and NR for Hybrid+KG+CSCG (no GNN) | Ablation configurations | Table VII, PDF p. 8 |
| P@10 significance vs baselines | Wilcoxon: Std RAG p<0.0001, d=1.41; Self-RAG p<0.0001, d=1.19; CRAG p<0.0001, d=0.98; GraphRAG p=0.0003, d=0.81; Agentic RAG p=0.0071, d=0.63 | Table says n=500 queries; corrected α=0.01 | Table VIII, PDF p. 8 |
| End-to-end latency | ResearchRAG mean 18.7 s/synthesis query; Standard RAG 2.3 s | Hardware/software setup in §IV.C | Limitations §VIII, PDF p. 8 |
| Knowledge graph construction cost | 4.2 hours; 18 GB storage for 42,500-paper corpus; incremental update ≈0.8 s/document | 42,500 papers | Limitations §VIII, PDF p. 8 |
| Latency/error samples | >20 s for 31/150 queries (20.7%); three CSCG iterations on >5 sentences: 38.4 s mean | 150 stratified sampled queries | §VI.D, PDF p. 7 |

**Internal reporting inconsistency:** the abstract says 2,500 queries total; §IV.A describes 50 TREC-COVID topics, 500 SCIDOCS clusters and 250 custom gap queries; Table VIII says n=500 queries for P@10 significance; the gap-rating table is described on 250 custom queries. The 150-query error sample and 50-query gap-failure subsection are separate samples. These descriptions should not be collapsed into one common denominator. I report each figure at the paper’s stated location and do not validate its raw evaluation data. The publication does not provide enough raw run outputs here for independent verification.

## 6. Cross-system comparison table

| Evaluation dimension | HySemRAG | ResearchRAG | EWRGV |
|---|---|---|---|
| Gap identification metric | Gap trends described in an application; numerical gap quality: NR | Expert Likert composite 4.04/5 on 250 custom queries; Spearman ρ=0.71 vs 0.48 heuristic (preprint-reported) | Not Measured; saved LLM artifact has 6 candidates without labels |
| Validation metric | QA single-pass 68.3%; citation accuracy 99.0% ±4.5% in 394 verified observations; definitions differ from EWRGV classes | Hallucination rate 0.078 on synthesis; not EWRGV validation classes | Not Measured; no labeled result set |
| Retrieval metric | Structured-field vs chunk semantic similarity (not P/R); conventional P/R/MRR/nDCG: NR in reported result section | TREC-COVID P@10 0.831; R@10 0.709; MRR 0.774; nDCG@10 0.748 | Not Measured; evaluator labels absent |
| Human/expert evaluation | No expert gap-quality score identified in reported results (NR) | Two PhD annotators; Likert ratings; κ=0.67; SCIDOCS synthesis references κ=0.69 | Not Measured |
| Dataset and sample size | Development evaluation 643 observations / 60 sessions; application 34 studies | TREC-COVID, SCIDOCS and custom gap queries; sample descriptions conflict as detailed above | No documented evaluation dataset; one unlabelled artifact |
| Runtime/cost | NR in inspected evaluation results | 18.7 s/query vs 2.3 s baseline; graph build 4.2 h/42,500 papers | Not Measured; no live run |

**Comparability:** no result above is a direct head-to-head comparison with EWRGV. EWRGV has no benchmark data. The systems use different datasets, query/gap definitions, output tasks, labels, annotators and protocols; HySemRAG’s similarity and citation measures do not equal ResearchRAG’s retrieval metrics or EWRGV’s VALID/UNCERTAIN/UNSUPPORTED classification. ResearchRAG’s own sample descriptions require clarification. No superiority conclusion is supported.

## 7. Limitations and fairness assessment

- There is no common query set, paper corpus, candidate-gap label set, or annotation procedure across the three systems.
- EWRGV's current code scores evidence and classifies candidates, but its current repository contains no reference outcomes for computing classification or ranking quality. Model confidence cannot substitute for ground truth.
- The only saved EWRGV candidate artifact is an LLM-output run, not an evaluation sample; it does not establish signal-detector performance or validation performance.
- HySemRAG reports controlled developer testing, has ambiguously reconciled denominators, and reports semantic similarity not directly comparable to standard retrieval ranking.
- ResearchRAG is a preprint with published claims but no raw result bundle available for independent reproduction in this workspace; its stated query/sample totals differ by section.
- Published runtime is hardware-dependent. ResearchRAG reports an A100 server setup; EWRGV has no measured runtime on a specified matching environment.
- Unit tests establish tested code behavior under the tested fixtures; they do not establish retrieval relevance, research-gap correctness, live API availability, or full pipeline success.

## 8. Recommended next experiments

1. **Define a fixed shared evaluation set first.** Use a versioned query set and paper corpus, with expert-labeled relevant papers/passages and candidate gap judgments. For gap validation, label whether each proposed gap is open, uncertain, or already addressed, document adjudication, and avoid training/evaluating on the same examples.
2. **Retrieval benchmark.** On the same queries/corpus, record pre-rerank and post-rerank ranked passage IDs and judgments. Compute Precision@5/10, Recall@5/10/20, MRR, MAP and nDCG@10 with one explicit relevance rule. Save raw rankings and script outputs.
3. **Gap candidate benchmark.** Freeze signal detectors/configuration and record generated candidates, detector family, evidence, rank and filtering decisions. Match candidates to expert-labeled gaps under a documented matching rule; compute precision/recall/F1 and ranking metrics, with per-signal-family counts and retained proportions.
4. **Validation benchmark.** Run frozen candidates against independent reference labels. Report confusion matrix and per-class/overall precision, recall, F1 and accuracy; report confidence distribution and calibration (e.g. Brier score/reliability curve) only if the confidence is intended as a probability and calibration labels are available. Count evidence types, searches, unavailable providers, failures and per-candidate latency.
5. **Repeatable systems run.** On a fixed corpus and pinned model/provider versions, log papers collected/deduplicated/downloaded/parsed/chunked, stage timings, API outcomes, end-to-end timings, hardware and cost. Report failed/partial runs and do not silently omit them.
6. **Fair published-baseline reproduction.** Obtain/verify both papers’ benchmark splits, annotations, scripts and raw outputs. If exact systems are reproducible, evaluate EWRGV and baselines on the same inputs and labels; otherwise report a new common benchmark separately from the published tables. Resolve ResearchRAG sample-count inconsistencies before interpreting significance tests.

Minimum realistic first step in this repository: assemble a small expert-labeled set and benchmark sparse/dense/hybrid retrieval plus reranking on a frozen local corpus; then label gap outcomes for that same query set. Without labels, only operational counts and timings—not quality scores—can be honestly reported.

## 9. Evidence appendix

### Repository files and paths

- Project: `C:\Users\Muthu\Downloads\Refine-ResearchGap\ewrgv`
- Core docs: `README.md`, `docs/SYSTEM_TEST_GUIDE.md`, `docs/retrieval_api.md`, `docs/gaps_kg_api.md`, `docs/audit/CODEBASE_AUDIT.md`, `docs/audit/IMPLEMENTATION_STATUS.md`.
- Evaluation-related scripts: `scripts/verify_retrieval.py` (prints sparse/dense/hybrid results; no labeled metric computation), `scripts/verify_deduplication.py` (requires configured DB; operational counts, not quality metrics), `scripts/run_llm_gaps.py`, `scripts/run_llm_gap_baseline.py`.
- Relevant tests: `tests/unit/test_reranking.py`, `test_candidate_ranker.py`, `test_candidate_evidence_scoring.py`, plus other retrieval/gap/validation unit and integration tests.
- Raw artifacts: `data/artifacts/bm25/*/bm25_index.pkl`; `data/artifacts/gaps/e0b54bcc-ebcd-4e77-ae6e-d8e4fb4ef90d_llm_gaps.json`.
- Papers supplied: `C:\Users\Muthu\Downloads\Refine-ResearchGap\initial-docs\HySemRAG.pdf`; `C:\Users\Muthu\Downloads\Refine-ResearchGap\initial-docs\ResearchRAG_A_Knowledge-Graph-Enhanced_Hybrid_Mult.pdf`.

### Commands run

- `python -m pytest tests/unit/test_reranking.py tests/unit/test_candidate_ranker.py tests/unit/test_candidate_evidence_scoring.py -q` — 113 passed in 1.16 s.
- `python -m pytest tests -q` — attempted, did not complete; interrupted without a final test summary.
- `python -m pytest tests/unit/test_reranking.py tests/unit/test_candidate_ranker.py tests/unit/test_candidate_evidence_scoring.py tests/unit/test_gaps_kg_api.py -q` — partial output, stalled before final summary; interrupted.
- Read-only PDF text extraction via installed PyMuPDF for both supplied PDFs; the report’s page references are PDF page numbers (printed page numbers align for the cited results pages).
- Web verification: [arXiv primary record for HySemRAG](https://arxiv.org/abs/2508.05666); [Research Square DOI landing URL](https://doi.org/10.21203/rs.3.rs-10028443/v1) was used as the stable publication identifier printed in the supplied ResearchRAG PDF. The local preprint supplied the detailed metric tables and page locations.

### Formulas / interpretation

- No EWRGV quality metric was calculated: the necessary human relevance or gap labels are absent. No model output was treated as a label.
- The saved artifact counts are direct list lengths (`len(retrieved_chunks)=20`, `len(candidate_gaps)=6`, `len(top_k_gaps)=5`); they describe that artifact only.
- Published baseline figures are transcribed as the papers label them. No conversion from similarity, Likert rating, acceptance, or citation accuracy into precision/recall/accuracy was performed.
- The 113/113 unit-test pass fraction is 113 ÷ 113 = 100% for the named test command only; it is not a model-quality metric.

### Reproducibility status

- EWRGV unit-level logic for reranking and candidate scoring is reproducible under the recorded command. Its research-quality performance is not reproducible from the current repository because the required labeled set and saved predictions are missing.
- The documented real pipeline requires the configured DB/storage and providers and was not run here. No claims of end-to-end success are made.
- HySemRAG and ResearchRAG reported results are not independently reproduced. ResearchRAG’s DOI and paper identity are taken from the supplied preprint; its benchmark counts and raw outputs could not be independently checked from an official evaluation artifact in this repository.

### Metric definitions and calculation status

For EWRGV, the standard retrieval metrics would require a fixed query set and relevance judgments: Precision@K = relevant retrieved items in the first K divided by K; Recall@K = relevant retrieved items in the first K divided by all relevant items for the query; reciprocal rank = 1/rank of the first relevant item (MRR is its mean across queries); AP averages precision at ranks containing relevant items (MAP is the mean AP); and nDCG discounts graded relevance by rank and normalizes against ideal ordering. The repository has neither the required judgments nor an evaluation routine using them, so these are not computable from the current recorded artifacts.

Candidate precision/recall/F1, validation classification metrics/confusion matrix, and confidence calibration likewise require independent human reference labels and frozen model outputs; neither is available. Directly countable operational quantities in an artifact (retrieved chunks/candidates) do not estimate these quality metrics. The reranker’s elapsed-millisecond return value is a per-call instrumentation hook; without a recorded run, environment, workload and aggregation rule it is not a measured latency result.

Published baseline metric names and values above are left in each paper’s terminology. Where a paper does not report an EWRGV-equivalent metric—such as EWRGV’s three-way validation class accuracy—it is NR, not inferred from a different score.
