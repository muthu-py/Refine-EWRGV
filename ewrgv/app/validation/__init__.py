"""
app/validation/__init__.py
---------------------------
Evidence-Weighted Research Gap Validation (EWRGV) package.

This is the core research contribution of the EWRGV system.

Responsibilities
----------------
* Receive Top-K CandidateGap objects from gap_detection/.
* Run multi-dimensional validation searches (direct, terminology, method,
  counter) to expand evidence beyond the initial retrieval.
* Classify each collected evidence item.
* Analyze coverage across multiple dimensions.
* Aggregate evidence.
* Compute a confidence score.
* Produce a final GapClassification: VALID_GAP | UNCERTAIN_GAP | UNSUPPORTED_GAP.
* Generate an evidence-chain explanation.

Architectural note
------------------
This package is deliberately separate from gap_detection/.
The research boundary is:

    Gap Detection  (gap_detection/)
          ↓
    CandidateGap   (domain model)
          ↓
    EWRGV Validation  (validation/)   ← THIS PACKAGE
          ↓
    ValidationResult  (domain model)
"""
