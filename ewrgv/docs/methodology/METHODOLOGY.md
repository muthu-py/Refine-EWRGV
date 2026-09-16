# EWRGV Methodology Overview

## Evidence-Weighted Research Gap Validation

This document summarises the research methodology that the EWRGV system implements.

---

## Core Distinction

```
Gap Detection  ≠  Gap Validation
```

Existing RAG-based gap-identification approaches stop at detection.  EWRGV adds a dedicated validation phase that produces calibrated, evidence-backed classifications.

---

## Pipeline Overview

```
Research Question
        ↓
Query Understanding
        ↓
Query Expansion
        ↓
Literature Collection
        ↓
Document Processing
        ↓
Literature Corpus
        ↓
Hybrid Retrieval
        ↓
Reranking
        ↓
Evidence Collection
        ↓
Gap Identification
        ↓
Candidate Research Gaps
        ↓
Candidate Ranking
        ↓
Top-K Candidate Gaps
        ↓
EWRGV Validation
        ↓
Evidence Classification
        ↓
Coverage Analysis
        ↓
Evidence Aggregation
        ↓
Confidence Assessment
        ↓
Final Classification
        ↓
Evidence-Chain Explanation
```

---

## EWRGV Validation Steps

1. **Generate Verification Queries** — from the structured gap representation
2. **Direct Evidence Search** — verbatim gap concept retrieval
3. **Terminology Search** — synonym and alternative-term expansion
4. **Method Variation Search** — alternative methodological framings
5. **Counter-Evidence Search** — mandatory search for refuting evidence
6. **Evidence Classification** — per-item: SUPPORTING | COUNTER | PARTIAL | CONTRADICTORY | INSUFFICIENT
7. **Coverage Analysis** — multi-dimensional: retrieval, concept, terminology, method, corpus
8. **Evidence Aggregation** — de-duplication and weighting
9. **Confidence Assessment** — formula combining all dimensions (TBD)
10. **Final Classification** — VALID_GAP | UNCERTAIN_GAP | UNSUPPORTED_GAP
11. **Evidence-Chain Explanation** — step-by-step narrative

---

## Gap Taxonomy

| Type | Description |
|---|---|
| Methodological | No adequate method exists for the problem |
| Population | A specific population is understudied |
| Dataset | No suitable dataset exists |
| Geographic | Geographic regions are underrepresented |
| Temporal | Time periods are underrepresented |
| Contradictory Evidence | Conflicting findings exist without resolution |
| Evaluation | Lack of evaluation frameworks or benchmarks |
| Other / Unknown | Does not fit standard categories |

---

## Evidence Types

| Type | Meaning |
|---|---|
| SUPPORTING | Confirms the gap exists (no one has solved it) |
| COUNTER | Suggests the gap is already addressed |
| PARTIAL | Partially addresses the gap |
| CONTRADICTORY | Directly contradicts the gap claim |
| INSUFFICIENT | Related but too weak to classify meaningfully |

---

## Confidence Thresholds (configurable)

| Score Range | Classification |
|---|---|
| ≥ 0.70 | VALID_GAP |
| ≥ 0.40 | UNCERTAIN_GAP |
| < 0.40 | UNSUPPORTED_GAP |
