"""
app/validation/counter_search.py
----------------------------------
Mandatory counter-evidence search.

The system must actively look for work that already addresses the gap:
queries are phrased as if the gap were solved ("<concepts> novel framework",
"<concepts> proposed method improves").  Skipping this step would let a gap
"exist" only because the initial retrieval never surfaced the work that
fills it.
"""

from __future__ import annotations

from app.validation.direct_search import ValidationSearch


class CounterEvidenceSearch(ValidationSearch):
    search_type = "counter"
