"""
app/core/logging.py
-------------------
Centralised structured logging configuration for the EWRGV application.

Uses Python's standard `logging` module with a consistent format so that
every module gets a properly-named logger without repeating configuration.

Usage
-----
    from app.core.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Pipeline started", extra={"query": query_text})
"""

from __future__ import annotations

import logging
import sys
from typing import Optional


_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def configure_logging(level: str = "INFO") -> None:
    """
    Configure the root logger.

    Call this once at application startup (e.g. in main.py lifespan).
    Subsequent calls are idempotent because we check for existing handlers.
    """
    root = logging.getLogger()

    if root.handlers:
        # Already configured – avoid adding duplicate handlers.
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a named logger.

    Parameters
    ----------
    name:
        Typically pass ``__name__`` from the calling module so that the
        logger hierarchy mirrors the package hierarchy.
    """
    return logging.getLogger(name or "ewrgv")
