"""
app/gap_detection/__init__.py
------------------------------
Gap detection package.

Responsibilities
----------------
* Detect candidate research gaps from retrieved evidence chunks.
* Produce structured gap representations.
* Rank candidates to produce the Top-K gaps for EWRGV validation.

Output:  list[CandidateGap]  →  passed to the validation/ package.

IMPORTANT: This package does NOT validate gaps.
Gap detection and gap validation are intentionally separated.
"""
