"""
tests/unit/test_stub_corpus.py
--------------------------------
Unit tests for StubCorpusStore.

Verifies:
    - Deterministic corpus returned
    - Correct paper and chunk counts
    - Retrieve chunks by paper_id
    - Paper lookup by ID
    - Unknown IDs return empty/None
"""

from __future__ import annotations

from app.stubs.stub_corpus import StubCorpusStore


class TestStubCorpusStore:
    """Tests for the deterministic mock corpus store."""

    def setup_method(self) -> None:
        self.store = StubCorpusStore()

    # ------------------------------------------------------------------ #
    # Determinism
    # ------------------------------------------------------------------ #

    def test_returns_deterministic_papers(self) -> None:
        """Two calls return the same papers in the same order."""
        papers1 = self.store.get_all_papers()
        papers2 = self.store.get_all_papers()
        assert len(papers1) == len(papers2)
        for p1, p2 in zip(papers1, papers2):
            assert p1.paper_id == p2.paper_id
            assert p1.title == p2.title

    def test_returns_deterministic_chunks(self) -> None:
        """Two calls return the same chunks in the same order."""
        chunks1 = self.store.get_all_chunks()
        chunks2 = self.store.get_all_chunks()
        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1.chunk_id == c2.chunk_id
            assert c1.text == c2.text

    # ------------------------------------------------------------------ #
    # Counts
    # ------------------------------------------------------------------ #

    def test_paper_count(self) -> None:
        """Corpus contains exactly 5 papers."""
        assert len(self.store.get_all_papers()) == 5

    def test_chunk_count(self) -> None:
        """Corpus contains exactly 13 chunks."""
        assert len(self.store.get_all_chunks()) == 13

    # ------------------------------------------------------------------ #
    # Lookup by ID
    # ------------------------------------------------------------------ #

    def test_get_paper_by_id_found(self) -> None:
        """Known paper ID returns the correct paper."""
        paper = self.store.get_paper_by_id("paper-001")
        assert paper is not None
        assert paper.paper_id == "paper-001"
        assert "Attention" in paper.title

    def test_get_paper_by_id_not_found(self) -> None:
        """Unknown paper ID returns None."""
        assert self.store.get_paper_by_id("nonexistent") is None

    def test_get_chunks_by_paper_id(self) -> None:
        """Retrieves chunks for a specific paper."""
        chunks = self.store.get_chunks_by_paper_id("paper-001")
        assert len(chunks) == 3
        for c in chunks:
            assert c.paper_id == "paper-001"

    def test_get_chunks_by_paper_id_unknown(self) -> None:
        """Unknown paper ID returns an empty list."""
        assert self.store.get_chunks_by_paper_id("nonexistent") == []

    def test_get_chunk_by_id(self) -> None:
        """Known chunk ID returns the correct chunk."""
        chunk = self.store.get_chunk_by_id("chunk-001-intro")
        assert chunk is not None
        assert chunk.chunk_id == "chunk-001-intro"
        assert chunk.paper_id == "paper-001"

    def test_get_chunk_by_id_not_found(self) -> None:
        """Unknown chunk ID returns None."""
        assert self.store.get_chunk_by_id("nonexistent") is None

    # ------------------------------------------------------------------ #
    # Data integrity
    # ------------------------------------------------------------------ #

    def test_all_chunks_reference_valid_papers(self) -> None:
        """Every chunk's paper_id corresponds to a real paper."""
        paper_ids = {p.paper_id for p in self.store.get_all_papers()}
        for chunk in self.store.get_all_chunks():
            assert chunk.paper_id in paper_ids

    def test_chunks_have_nonempty_text(self) -> None:
        """Every chunk has non-empty text content."""
        for chunk in self.store.get_all_chunks():
            assert chunk.text.strip() != ""

    def test_papers_have_nonempty_titles(self) -> None:
        """Every paper has a non-empty title."""
        for paper in self.store.get_all_papers():
            assert paper.title.strip() != ""
