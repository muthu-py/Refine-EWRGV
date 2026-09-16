"""
app/query_understanding/__init__.py
-------------------------------------
Query Understanding & Expansion package (Phase 2.1).

Exports the public surface of the module.

    from app.query_understanding import QueryUnderstandingService

The service is the only entry point for application and pipeline code.
Internal schemas and prompt helpers are intentionally not re-exported here
to keep the package boundary clean.
"""

from app.query_understanding.service import QueryUnderstandingService

__all__ = ["QueryUnderstandingService"]
