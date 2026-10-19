"""
app/coverage/corpus.py - corpus adequacy.

Is there enough literature behind this judgement?  ResearchRAG's error
analysis found 24% of its gaps were artefacts of an incomplete corpus
(fields with < 30 indexed papers).  Adequacy combines

    run papers mentioning the gap's concepts      min(1, n_run / 10)
    relevant external items found                 min(1, n_ext / 15)

    adequacy    = 0.5 * run part + 0.5 * external part
    uncertainty = 1 - adequacy

The confidence scorer caps confidence when adequacy is low (see
app/scoring/confidence.py).
"""

from __future__ import annotations


def compute_corpus_adequacy(run_papers: int, relevant_external: int) -> float:
    return round(0.5 * min(1.0, run_papers / 10) + 0.5 * min(1.0, relevant_external / 15), 4)


def compute_corpus_uncertainty(adequacy: float | None) -> float | None:
    return None if adequacy is None else round(1.0 - adequacy, 4)
