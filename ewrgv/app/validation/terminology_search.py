"""
app/validation/terminology_search.py
--------------------------------------
Terminology search: the gap's concepts replaced by knowledge-graph aliases
and synonyms, so a gap is not declared just because the literature uses
other words for it.
"""

from __future__ import annotations

from app.validation.direct_search import ValidationSearch


class TerminologySearch(ValidationSearch):
    search_type = "terminology"
