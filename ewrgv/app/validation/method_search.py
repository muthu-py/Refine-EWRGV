"""
app/validation/method_search.py
---------------------------------
Method-variation search: a METHOD / DATASET / MODALITY concept replaced by a
sibling of the same type that co-occurs with the gap's other concepts in the
knowledge graph (e.g. "vision transformer" -> "swin transformer").
"""

from __future__ import annotations

from app.validation.direct_search import ValidationSearch


class MethodVariationSearch(ValidationSearch):
    search_type = "method_variation"
