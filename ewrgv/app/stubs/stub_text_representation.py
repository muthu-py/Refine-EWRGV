"""
app/stubs/stub_text_representation.py
---------------------------------------
Deterministic text representation stubs for development and testing.

.. warning::
    These are **development stubs**.  They will be replaced by:
    - Real embedding API (OpenAI, Sentence-Transformers) for dense vectors
    - Real tokenization pipeline for sparse/lexical representations

Provides:
    StubEmbeddingProvider — deterministic 128-dim embeddings via hashing
    StubLexicalRepresenter — deterministic tokenisation + normalisation

Both implementations are fully deterministic: the same input always
produces the same output, with no randomness or external API calls.
"""

from __future__ import annotations

import hashlib
import math
import re


# ======================================================================
# Dense / Vector Representation Stub
# ======================================================================

# Fixed dimensionality for all stub embeddings.
EMBEDDING_DIM = 128


class StubEmbeddingProvider:
    """
    Deterministic embedding provider that generates fixed-dimension vectors.

    Uses a hash-based approach: the input text is SHA-256 hashed, and the
    hash bytes are expanded into a float vector of ``EMBEDDING_DIM``
    dimensions.  The vector is then L2-normalised to unit length.

    Satisfies the ``EmbeddingProvider`` protocol (app.domain.interfaces).

    Properties:
        - Same input text → same output vector (always)
        - Different input texts → different vectors (with very high probability)
        - All vectors have exactly ``EMBEDDING_DIM`` dimensions
        - All vectors are L2-normalised (unit length)
        - No external API calls

    .. warning::
        **Development stub.**  Replace with a real embedding model in production.
    """

    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        """The dimensionality of the embedding vectors."""
        return self._dim

    def embed(self, text: str) -> list[float]:
        """Return a deterministic dense vector for a single text string."""
        return self._hash_to_vector(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for a batch of texts."""
        return [self._hash_to_vector(t) for t in texts]

    def _hash_to_vector(self, text: str) -> list[float]:
        """
        Convert text to a deterministic unit-length float vector.

        Strategy:
        1. SHA-256 hash the text (32 bytes).
        2. Repeat the hash bytes to fill ``_dim`` slots.
        3. Convert each byte to a float in [-1, 1].
        4. L2-normalise the vector.
        """
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Expand to dim values by cycling through digest bytes
        raw = []
        for i in range(self._dim):
            byte_val = digest[i % len(digest)]
            # Map 0–255 to -1.0–1.0
            raw.append((byte_val / 127.5) - 1.0)

        # L2 normalise
        norm = math.sqrt(sum(x * x for x in raw))
        if norm > 0:
            raw = [x / norm for x in raw]
        return raw


# ======================================================================
# Sparse / Lexical Representation Stub
# ======================================================================

# Minimal English stop-word set for normalisation.
_STOP_WORDS: set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very",
    "and", "but", "or", "if", "while", "because", "until", "about",
    "this", "that", "these", "those", "it", "its", "he", "she", "they",
    "we", "you", "i", "me", "my", "your", "his", "her", "their", "our",
    "which", "who", "whom", "what",
}


class StubLexicalRepresenter:
    """
    Deterministic lexical tokenisation and normalisation for BM25-style retrieval.

    Processing steps:
    1. Lowercase the input text.
    2. Remove non-alphanumeric characters (keep spaces).
    3. Split on whitespace.
    4. Remove stop words.
    5. Filter tokens shorter than 2 characters.

    No stemming or lemmatisation is applied — this is a deliberate
    simplification for the stub.

    .. warning::
        **Development stub.**  Replace with a proper NLP tokenisation
        pipeline in production.
    """

    def tokenize(self, text: str) -> list[str]:
        """
        Return a list of normalised tokens from the input text.

        The result is deterministic for the same input.
        """
        # Lowercase
        text = text.lower()
        # Remove non-alphanumeric (keep spaces)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        # Split and filter
        tokens = text.split()
        tokens = [t for t in tokens if t not in _STOP_WORDS and len(t) >= 2]
        return tokens
