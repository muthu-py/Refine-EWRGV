"""
app/retrieval/bm25_index.py
---------------------------
Sparse representation implementation using rank-bm25.

Provides deterministic index building and local persistence for Phase 2.9.
"""

from __future__ import annotations

import pickle
import re
import unicodedata
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from app.core.logging import get_logger
from app.domain.models.paper_chunk import PaperChunk

logger = get_logger(__name__)

# Minimal, conservative stopword list.
# We do not aggressively remove words that might be important in academic texts.
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for",
    "if", "in", "into", "is", "it", "no", "not", "of", "on", "or",
    "such", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with"
}

# Regex for tokenization. Keeps alphanumeric blocks, handles common technical terms like GPT-4, COVID-19.
_TOKEN_PATTERN = re.compile(r"(?u)\b\w+(?:-\w+)*\b")


def preprocess_text(text: str) -> list[str]:
    """
    Deterministic text preprocessing for BM25.

    Steps:
    1. Unicode NFC normalization
    2. Lowercase
    3. Tokenization (preserves hyphens within words)
    4. Minimal stop-word removal

    Parameters
    ----------
    text:
        Raw text to preprocess.

    Returns
    -------
    list[str]:
        List of processed tokens.
    """
    # 1. Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # 2. Lowercase
    text = text.lower()

    # 3. Tokenize
    tokens = _TOKEN_PATTERN.findall(text)

    # 4. Stop-word removal
    return [t for t in tokens if t not in _STOP_WORDS]


class BM25Index:
    """
    Wraps rank_bm25 for building, querying, and persisting the sparse index.
    """

    def __init__(self) -> None:
        self._bm25: Optional[BM25Okapi] = None
        self._chunk_ids: list[str] = []

    def build(self, chunks: list[PaperChunk]) -> "BM25Index":
        """
        Build the BM25 index from a list of chunks.

        Parameters
        ----------
        chunks:
            The canonical chunks to index.

        Returns
        -------
        BM25Index:
            Self, for chaining.
        """
        if not chunks:
            logger.warning("BM25Index.build() called with empty chunk list.")
            self._bm25 = None
            self._chunk_ids = []
            return self

        # Extract IDs and preprocess text
        self._chunk_ids = [c.chunk_id for c in chunks]
        tokenized_corpus = [preprocess_text(c.text) for c in chunks]

        # Build index
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info("Built BM25 index with %d chunks.", len(self._chunk_ids))
        return self

    def query(self, text: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Query the index and return top chunk IDs with scores.

        Parameters
        ----------
        text:
            The search query.
        top_k:
            Maximum number of results to return.

        Returns
        -------
        list[tuple[str, float]]:
            List of (chunk_id, score) tuples, sorted by descending score.
        """
        if self._bm25 is None or not self._chunk_ids:
            return []

        tokenized_query = preprocess_text(text)
        if not tokenized_query:
            return []

        scores = self._bm25.get_scores(tokenized_query)

        # Get indices of top_k scores
        results = [(chunk_id, score) for chunk_id, score in zip(self._chunk_ids, scores) if score > 0]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def save(self, file_path: str | Path) -> None:
        """
        Persist the index to disk.

        Parameters
        ----------
        file_path:
            Path to save the pickle file.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "bm25": self._bm25,
            "chunk_ids": self._chunk_ids,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.debug("Saved BM25 index to %s", path)

    @classmethod
    def load(cls, file_path: str | Path) -> "BM25Index":
        """
        Load a persisted index from disk.

        Parameters
        ----------
        file_path:
            Path to the pickle file.

        Returns
        -------
        BM25Index:
            Loaded index instance.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"BM25 index file not found: {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)

        instance = cls()
        instance._bm25 = data["bm25"]
        instance._chunk_ids = data["chunk_ids"]
        logger.debug("Loaded BM25 index from %s (%d chunks)", path, len(instance._chunk_ids))
        return instance

    @property
    def chunk_ids(self) -> list[str]:
        """IDs of the indexed chunks, in index order."""
        return list(self._chunk_ids)

    @property
    def size(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._chunk_ids)

    def is_empty(self) -> bool:
        """Return True if the index is empty."""
        return self._bm25 is None or not self._chunk_ids
