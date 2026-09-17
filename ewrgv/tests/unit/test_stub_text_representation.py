"""
tests/unit/test_stub_text_representation.py
---------------------------------------------
Unit tests for StubEmbeddingProvider and StubLexicalRepresenter.

Verifies:
    - Deterministic embeddings (same input → same output)
    - Fixed vector dimensions
    - Batch embedding consistency
    - Lexical tokenisation produces expected tokens
    - Stop words are removed
"""

from __future__ import annotations

from app.stubs.stub_text_representation import (
    EMBEDDING_DIM,
    StubEmbeddingProvider,
    StubLexicalRepresenter,
)


class TestStubEmbeddingProvider:
    """Tests for the deterministic hash-based embedding provider."""

    def setup_method(self) -> None:
        self.provider = StubEmbeddingProvider()

    # ------------------------------------------------------------------ #
    # Determinism
    # ------------------------------------------------------------------ #

    def test_same_input_same_output(self) -> None:
        """The same text always produces the same embedding."""
        text = "Transformer models for NLP"
        v1 = self.provider.embed(text)
        v2 = self.provider.embed(text)
        assert v1 == v2

    def test_different_inputs_different_outputs(self) -> None:
        """Different texts produce different embeddings."""
        v1 = self.provider.embed("Transformer")
        v2 = self.provider.embed("Convolutional")
        assert v1 != v2

    # ------------------------------------------------------------------ #
    # Dimensions
    # ------------------------------------------------------------------ #

    def test_embedding_dimension(self) -> None:
        """Embeddings have exactly EMBEDDING_DIM dimensions."""
        v = self.provider.embed("test text")
        assert len(v) == EMBEDDING_DIM

    def test_custom_dimension(self) -> None:
        """Custom dimension is respected."""
        p = StubEmbeddingProvider(dim=64)
        v = p.embed("test")
        assert len(v) == 64

    # ------------------------------------------------------------------ #
    # Normalisation
    # ------------------------------------------------------------------ #

    def test_unit_length(self) -> None:
        """Embeddings are approximately unit-length (L2 norm ≈ 1)."""
        import math
        v = self.provider.embed("normalization test")
        norm = math.sqrt(sum(x * x for x in v))
        assert abs(norm - 1.0) < 1e-6

    # ------------------------------------------------------------------ #
    # Batch
    # ------------------------------------------------------------------ #

    def test_batch_embedding(self) -> None:
        """Batch embedding produces the same results as individual calls."""
        texts = ["Transformer", "BERT", "GPT-3"]
        batch_results = self.provider.embed_batch(texts)
        assert len(batch_results) == 3
        for text, batch_vec in zip(texts, batch_results):
            individual_vec = self.provider.embed(text)
            assert batch_vec == individual_vec

    def test_batch_empty(self) -> None:
        """Empty batch returns empty list."""
        assert self.provider.embed_batch([]) == []

    # ------------------------------------------------------------------ #
    # Type checks
    # ------------------------------------------------------------------ #

    def test_returns_list_of_floats(self) -> None:
        """Embedding is a list of float values."""
        v = self.provider.embed("test")
        assert isinstance(v, list)
        assert all(isinstance(x, float) for x in v)


class TestStubLexicalRepresenter:
    """Tests for the deterministic lexical tokeniser."""

    def setup_method(self) -> None:
        self.tokenizer = StubLexicalRepresenter()

    # ------------------------------------------------------------------ #
    # Determinism
    # ------------------------------------------------------------------ #

    def test_same_input_same_output(self) -> None:
        """Same text always produces the same tokens."""
        text = "The Transformer model uses self-attention"
        t1 = self.tokenizer.tokenize(text)
        t2 = self.tokenizer.tokenize(text)
        assert t1 == t2

    # ------------------------------------------------------------------ #
    # Normalisation
    # ------------------------------------------------------------------ #

    def test_lowercase(self) -> None:
        """All tokens are lowercase."""
        tokens = self.tokenizer.tokenize("BERT GPT Transformer")
        for t in tokens:
            assert t == t.lower()

    def test_stop_words_removed(self) -> None:
        """Common stop words are excluded."""
        tokens = self.tokenizer.tokenize("the model is very good")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "very" not in tokens

    def test_short_tokens_removed(self) -> None:
        """Single-character tokens are excluded."""
        tokens = self.tokenizer.tokenize("a b c model")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "c" not in tokens

    def test_punctuation_removed(self) -> None:
        """Punctuation is stripped from tokens."""
        tokens = self.tokenizer.tokenize("model, architecture; results.")
        assert "model" in tokens
        assert "architecture" in tokens
        assert "results" in tokens
        # No punctuation tokens
        assert "," not in tokens
        assert ";" not in tokens

    # ------------------------------------------------------------------ #
    # Content
    # ------------------------------------------------------------------ #

    def test_meaningful_tokens_preserved(self) -> None:
        """Meaningful content words are preserved."""
        tokens = self.tokenizer.tokenize(
            "Transformer architecture for natural language processing"
        )
        assert "transformer" in tokens
        assert "architecture" in tokens
        assert "natural" in tokens
        assert "language" in tokens
        assert "processing" in tokens

    def test_empty_input(self) -> None:
        """Empty input returns empty list."""
        assert self.tokenizer.tokenize("") == []

    def test_only_stopwords(self) -> None:
        """Input with only stop words returns empty list."""
        assert self.tokenizer.tokenize("the a is are") == []
