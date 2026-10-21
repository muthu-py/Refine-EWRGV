# EWRGV Labeled Benchmark Evaluation

**Run date:** 2026-10-20  
**Repository:** `Refine-ResearchGap`  
**Status:** Partial benchmark execution. Results below distinguish completed measurements from incomplete runs.

## Summary

The official TREC-COVID Round 5 collection was evaluated with EWRGV BM25, including a Cross-Encoder rerank of the top 50 candidates per topic. SciFact dev evidence retrieval was evaluated with BM25, dense retrieval and hybrid RRF. Raw ranked IDs and per-query metrics were saved. TREC-COVID dense/hybrid retrieval did not complete: full-corpus embedding of the 191k-document release exceeded several minutes without producing rankings. SciFact reranking was not completed. No SciFact claim-verification classification was scored because its claim-support labels are not compatible with EWRGV's research-gap validation classes.

**Research-gap identification precision/recall/F1: Not Measured. EWRGV three-way validation accuracy and F1: Not Measured.** Neither benchmark provides the corresponding ground truth.

## 1. Sources, versions, licenses, and tasks

| Dataset | Source/version used | License and scope | Evaluation task |
|---|---|---|---|
| TREC-COVID | NIST final Round 5 topics, qrels and valid ID list; CORD-19 2020-07-16 metadata release. The NIST instructions identify the complete Round 5 package as the recommended final collection. [NIST TREC-COVID data](https://ir.nist.gov/trec-covid/data.html), [CORD-19 repository](https://github.com/allenai/cord19/blob/master/README.md) | CORD-19 metadata contains a per-record `license` field and source licenses vary; text is used locally and not redistributed here. Only official release metadata title and abstract were loaded, not the 3.66 GB full-text archive. | Search the query field of each topic against the official Round 5 document set; use cumulative graded qrels. |
| SciFact | Official AllenAI corpus and `claims_dev.jsonl` from the repository data release. [SciFact repository](https://github.com/allenai/scifact), [published schema](https://github.com/allenai/scifact/blob/master/doc/data.md), [license](https://github.com/allenai/scifact/blob/master/LICENSE.md) | Repository describes claims/annotations as CC BY 4.0, abstracts as ODC-By 1.0, and code as Apache-2.0. Evaluation uses only dev claims and the abstract corpus; train/test are not used. | Retrieve papers with annotated rationales for 188 evidence-bearing dev claims. SUPPORT/CONTRADICT are evidence stance labels, not gap-validation labels. |

TREC-COVID files were inspected before downloading. The exact files are `metadata-2020-07-16.csv`, `topics-rnd5.xml`, `qrels-covid_d5_j0.5-5.txt`, and `docids-rnd5.txt`. The official CORD-19 Round 5 metadata CSV is 269,219,095 bytes; the full release archive is 3,662,861,028 bytes and was not downloaded. The metadata adapter retained the 191,175 IDs in NIST's valid-ID list; 191,132 had nonempty title/abstract text, 43 had no usable text, and all 69,318 qrel records referenced valid release IDs. The adapter handles duplicate `cord_uid` rows by selecting the row with the most usable title/abstract text and keeps the original ID.

SciFact has 5,183 unique corpus document IDs and 300 dev claims. There are 188 dev claims with evidence annotations (124 SUPPORT and 64 CONTRADICT) and 112 without evidence, which are excluded from evidence-retrieval scoring. Annotated evidence references 182 unique corpus documents, with no missing evidence IDs. The corpus and claim IDs are retained.

Downloads and extracted inputs are under `ewrgv/data/raw/benchmark/` (ignored by Git). Output JSON is under `ewrgv/data/processed/benchmark_eval/` (also ignored by Git); these files contain queries, qrels, raw ranked document IDs and per-query metric values. Use the official download source for any redistribution; benchmark text is not copied into this report.

## 2. Implementation compatibility and adapters

Inspected the actual implementation in `ewrgv/app/retrieval/`, `ewrgv/app/evidence/classifier.py`, `ewrgv/app/scoring/confidence.py`, and `ewrgv/app/scoring/classification.py`.

- BM25: `BM25Index` plus the `SparseRetriever` interface. The benchmark runner constructs production `PaperChunk` objects from official document IDs and text, then uses `BM25Index` tokenization/scoring (rank-bm25 BM25Okapi; k1=1.5, b=0.75).
- Dense: `SentenceTransformerEmbeddingProvider` and the configured local `BAAI/bge-small-en-v1.5` model, 384 dimensions, normalized vectors; cosine similarity is dot product. For memory control, TREC dense indexing was implemented with batched model encoding into a float32 memmap. The approach preserves the configured model and cosine score, but the TREC run did not finish.
- Hybrid: production `reciprocal_rank_fusion_results`, k=60, combining the top 100 dense and top 100 BM25 results.
- Cross-Encoder: production `CrossEncoderReranker` using `cross-encoder/ms-marco-MiniLM-L-6-v2`, CPU, batch size 16; reranks the leading 50 results, retaining retrieval order after that window. The completed TREC rerank follows this setting.
- Data adapter: `ewrgv/scripts/benchmark_eval/datasets.py`. Evaluation code and metric tests are in `ewrgv/scripts/benchmark_eval/` and `ewrgv/tests/unit/test_benchmark_metrics.py`.

Environment model IDs were read from the local config; no paid API was called and no credentials were used. Exact upstream model revision hashes were not pinned in the local configuration, so weights are identified by model name rather than immutable revision. Inference was run offline using local caches. Retrieval depth=100, RRF k=60, rerank window=50, rerank batch=16; production algorithm logic/settings were not changed.

### Metric definitions

Binary relevance for P@5, P@10, R@10, MRR and MAP is qrel grade > 0. P denominators are fixed at k. AP divides by the number of qrel-relevant documents for the query. nDCG@10 uses graded gain `2^grade - 1` and log2 discount. Query metrics are macro-averaged. As in the default `trec_eval` convention, unjudged retrieved documents are treated as nonrelevant in the binary measures and as grade 0 for DCG; judged and unjudged retrieved counts are reported so this convention is visible. TREC-COVID residual pooling means this can penalize systems for retrieving unjudged documents and should not be read as proof of irrelevance.

## 3. TREC-COVID retrieval results

50 topics, 69,318 qrel entries, and 26,664 documents with grade > 0 across topic-specific qrels. `judged@10` and `unjudged@10` below are totals across 500 retrieved slots. Results are macro averages per topic. Only BM25 completed end-to-end; no dense or hybrid TREC score is available.

| Method | P@5 | P@10 | R@10 | MRR | MAP | nDCG@10 | judged@10 / unjudged@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25, before rerank | 0.6400 | 0.6220 | 0.0142 | 0.8207 | 0.0619 | 0.5424 | 434 / 66 |
| BM25 + Cross-Encoder, reranked top 50 | 0.7000 | 0.6660 | 0.0151 | 0.8337 | 0.0648 | 0.5851 | 436 / 64 |
| Dense | Not Measured — run incomplete | | | | | | |
| Hybrid RRF | Not Measured — requires completed dense run | | | | | | |

The reranker changed the top-10 ordering for the same BM25 candidate set; this is a benchmark observation, not an estimate of research-gap quality. Per-topic raw rankings and metrics are in `ewrgv/data/processed/benchmark_eval/trec_covid_round5_results.json`. The dense attempt embedded no complete TREC ranking set and therefore has no valid saved run or score; it was stopped after several minutes without completion.

## 4. SciFact evidence retrieval results

188 evidence-bearing dev claims are scored against their evidence-annotated document IDs. Evidence document relevance is binary (any annotated rationale for a document). Stance is deliberately ignored for retrieval relevance. The metric convention above applies; most retrieved documents are unjudged because SciFact evidence annotation is not exhaustive over the full corpus.

| Method | P@5 | P@10 | R@10 | MRR | MAP | nDCG@10 | judged@10 / unjudged@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BM25 | 0.1819 | 0.0968 | 0.9075 | 0.7769 | 0.7703 | 0.8035 | 182 / 1,698 |
| Dense BGE | 0.0543 | 0.0282 | 0.2628 | 0.2480 | 0.2427 | 0.2488 | 53 / 1,827 |
| Hybrid RRF (k=60) | 0.0628 | 0.0479 | 0.4569 | 0.3110 | 0.3001 | 0.3142 | 90 / 1,790 |
| Cross-Encoder reranking | Not Measured — run not completed | | | | | | |

The SciFact output with raw per-query rankings and qrels is `ewrgv/data/processed/benchmark_eval/scifact_dev_results.json`. The dense/hybrid metrics were run over the full 5,183-abstract corpus, not an evidence-only candidate subset. Reranking and classification were not scored. The repository's validation classifier makes EWRGV gap-status judgments from confidence and evidence scoring, while SciFact labels whether evidence supports or contradicts a scientific claim. Substituting SciFact stance prediction for EWRGV validation would change the task, so no EWRGV validation performance is claimed.

## 5. Research-gap identification and validation

No labels in the evaluated public datasets correspond directly to the meanings of EWRGV's `VALID_GAP`, `UNCERTAIN_GAP`, and `UNSUPPORTED_GAP` classes. TREC-COVID labels topic-document relevance (0/1/2). SciFact labels whether cited paper sentences SUPPORT or CONTRADICT a claim; it does not say whether a literature gap is real, uncertain, or unsupported. SciFact's 112 no-evidence claims also do not mean “unsupported research gap.”

| Requested measure | Status | Missing reference labels |
|---|---|---|
| Research-gap identification precision, recall, F1 | **Not Measured** | A frozen set of proposed gap statements with human judgments identifying actual gaps, non-gaps, and the matching evidence/scope. |
| EWRGV three-way validation accuracy/F1 | **Not Measured** | Reference status for each gap under EWRGV's operational definitions, including resolved/remaining gaps, uncertain cases, and unsupported candidates, plus independent adjudication. |
| SciFact claim-support classification | Not run as EWRGV validation | SciFact SUPPORT/CONTRADICT/NOINFO labels are claim-verification outcomes, not EWRGV gap classes. An NLI model would be a separate baseline and is not reported here. |

Potential gap-data lead: GAPMAP describes four datasets for explicit/implicit research-gap extraction, including explicit gap statements and scientific challenge/direction annotations ([paper](https://arxiv.org/abs/2510.25055), [repository](https://github.com/lhunter-lab/GAPMAP)). This schema could support extraction/span detection if files and annotations are verified, but it does not directly label EWRGV's three validation states. During inspection, the repository surfaced only a test text file in its Data view and its README license field was a placeholder; dataset-level provenance and redistribution terms were not verified. It is therefore **not recommended for immediate use as benchmark ground truth** until the actual files, annotation instructions, source provenance, and rights are independently confirmed. A 2026 Mendeley “Research Gap Discovery Dataset” record describes 3,326 arXiv papers with gaps automatically extracted by LLaMA-3.3/Mistral and gives CC BY 4.0. Those model-generated labels do not provide independent human ground truth, so that record was not used to score EWRGV.

## 6. Published comparison (not head-to-head)

The figures below are transcribed from the supplied paper review in `EWRGV_Evaluation_Metrics_Report.md`; they are published results, not reproduced in this experiment. Metric and dataset differences preclude direct ranking across systems.

| Evaluation dimension | EWRGV measurement here | HySemRAG published result | ResearchRAG published result |
|---|---|---|---|
| Retrieval | TREC-COVID BM25 P@10 0.622, R@10 0.014; BM25+reranker P@10 0.666, R@10 0.015. SciFact evidence retrieval results above. | Structured-field similarity 0.655±0.178 (n=165), chunking similarity 0.485±0.204 (n=263); these are not P/R retrieval metrics. | TREC-COVID P@10 0.831, R@10 0.709, MRR 0.774, nDCG@10 0.748. |
| Gap quality / rating | Not Measured; no compatible gap labels. | No standard gap P/R/F1 identified in the reviewed results. | Reports composite gap rating 4.04/5 vs Standard RAG 2.64, Agentic RAG 3.32, GraphRAG 3.27 and AutoSurvey 2.97 on custom queries. |
| Validation/classification | Not Measured. | Citation accuracy 99.0%±4.5% in its reported quality-controlled observations; not EWRGV gap-status accuracy. | Paper reports its own evaluation; no equivalent EWRGV label mapping. |

ResearchRAG comparisons above are reported as published; its sample descriptions and access to raw outputs are limitations already noted in the source review. Do not interpret the table as a controlled comparison.

## 7. Limits and reproducibility

- Domain and corpus limits: TREC-COVID is a 2020 COVID-19 collection and SciFact is a biomedical/scientific claim-verification benchmark. Neither represents broad interdisciplinary literature-gap discovery.
- Text coverage: TREC uses title+abstract metadata, not full text; 43 official IDs lacked usable title/abstract. SciFact uses abstracts. Full-text evidence, sections, and context are unavailable in these retrieval runs.
- Judgment incompleteness: TREC pools are residual and SciFact rationale annotation is not a complete document relevance judgment. Unjudged-as-zero scores can understate relevant retrieval; counts are exposed in outputs.
- TREC evaluation coverage: complete BM25 and BM25 reranking results; dense/hybrid incomplete. The complete official collection was not falsely represented as having a completed end-to-end evaluation.
- SciFact coverage: BM25, dense and RRF retrieval complete; cross-encoder reranking and claim classification not run. Only dev split was used; no train labels informed inference.
- No production algorithm/settings were changed. Added benchmark-only adapters, evaluator, runner, and metric tests.
- Model identifiers and settings are recorded above. The exact model repository commit hashes were not captured, so strict immutable model-weight reproduction requires recording those revisions in a follow-up run.
- Metric tests: `python -m pytest tests/unit/test_benchmark_metrics.py -q` — **2 passed**. They verify ranking metrics, duplicate handling, zero-relevant queries, macro aggregation, and unjudged counts.

### Reproduce completed runs

From `ewrgv/`, with the same cached local model IDs/configuration:

```powershell
$env:PYTHONPATH = (Resolve-Path .).Path
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
python -m pytest tests\unit\test_benchmark_metrics.py -q
python scripts\benchmark_eval\run_eval.py --dataset trec_covid_round5 --sparse-only
python scripts\benchmark_eval\run_eval.py --dataset scifact_dev --skip-rerank
```

The TREC command above reproduces BM25 plus reranking; its output JSON stores both pre- and post-rerank ranked IDs. The SciFact command reproduces BM25/dense/hybrid rankings without the uncompleted rerank step. Raw dataset inputs stay in `ewrgv/data/raw/benchmark/`.

## 8. Recommended next steps for gap-quality evaluation

1. Verify GAPMAP dataset file availability, per-dataset schemas, annotation instructions, provenance, and license before selecting a compatible gap-extraction subset.
2. Define EWRGV's gap status reference criteria with domain reviewers, including how multiple independent studies and contradictory evidence affect the three classes.
3. Freeze candidate generation and validation settings; have reviewers label a blinded, representative sample with adjudication and inter-annotator agreement.
4. Evaluate gap detection and three-way status separately, with per-domain precision/recall/F1, confusion matrices, calibration, and confidence intervals. Keep retrieval metrics separate from gap-quality metrics.
5. Complete TREC dense/hybrid with adequate compute and pin model revisions; complete SciFact reranking only if useful for the evidence-retrieval comparison. Do not train on either evaluation split.
