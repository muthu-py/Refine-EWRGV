"""
app/domain/interfaces.py
-------------------------
Abstract interfaces (Protocols) for all EWRGV system components.

Design rationale
----------------
* Every concrete implementation (LLM provider, vector store, retriever, etc.)
  must satisfy exactly one interface defined here.
* Application and orchestration code depends *only* on these interfaces, not
  on any concrete class.  This keeps infrastructure swappable without touching
  business logic.
* Python ``typing.Protocol`` (structural subtyping) is used so that concrete
  classes do not need to inherit from these protocols — they just need to
  implement the required methods.

Naming convention
-----------------
Interfaces are named after the role they play, not the technology behind them:
    LLMProvider, EmbeddingProvider, VectorStore, Retriever, ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol, runtime_checkable

from app.domain.enums import EvidenceType, GapClassification
from app.domain.models.candidate_gap import CandidateGap
from app.domain.models.evidence import Evidence
from app.domain.models.paper import DocumentChunk, Paper
from app.domain.models.query import ResearchQuery
from app.domain.models.validation import CoverageAssessment, ValidationResult


# ======================================================================
# Provider Interfaces
# ======================================================================


@runtime_checkable
class LLMProvider(Protocol):
    """
    Interface for any LLM backend (OpenAI, Anthropic, local GGUF, etc.).

    Application code calls ``complete`` and ``complete_structured`` via this
    interface; the concrete class handles the SDK details.
    """

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Return a text completion for the given prompt."""
        ...

    def complete_structured(self, prompt: str, schema: dict, **kwargs: Any) -> dict:
        """
        Return a JSON-structured completion that conforms to ``schema``.
        Callers should not depend on implementation details of function-calling.
        """
        ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Interface for text embedding models."""

    def embed(self, text: str) -> list[float]:
        """Return a dense vector embedding for a single text string."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for a batch of texts."""
        ...


# ======================================================================
# Literature Collection / Ingestion Interfaces
# ======================================================================


@runtime_checkable
class LiteratureCollector(Protocol):
    """
    Interface for fetching academic papers from external sources
    (Semantic Scholar, OpenAlex, arXiv, etc.).
    """

    def search(self, query: str, limit: int = 20) -> list[Paper]:
        """Search for papers matching the query string."""
        ...

    def fetch_by_id(self, paper_id: str) -> Optional[Paper]:
        """Fetch a single paper by its provider-specific identifier."""
        ...


@runtime_checkable
class DocumentParser(Protocol):
    """
    Interface for parsing raw document bytes / text into a Paper domain object.

    Implementations may wrap PyMuPDF, GROBID, or simple text parsers.
    """

    def parse(self, raw: bytes | str, metadata: dict) -> Paper:
        """
        Parse raw document content and return a structured Paper.

        Parameters
        ----------
        raw:
            Raw document content (PDF bytes or plain text).
        metadata:
            Externally-known metadata (title, authors, etc.) to merge in.
        """
        ...


# ======================================================================
# Retrieval Interfaces
# ======================================================================


@runtime_checkable
class Retriever(Protocol):
    """
    Interface for retrieving DocumentChunks relevant to a query string.

    Concrete implementations include:
        SemanticRetriever  – dense vector similarity search
        BM25Retriever      – sparse keyword search
        HybridRetriever    – fusion of the above
    """

    def retrieve(self, query: str, top_k: int = 20) -> list[DocumentChunk]:
        """Return the top-k most relevant chunks for the query."""
        ...


@runtime_checkable
class Reranker(Protocol):
    """Interface for reranking a list of retrieved chunks."""

    def rerank(
        self,
        query: str,
        chunks: list[DocumentChunk],
        top_n: int = 10,
    ) -> list[DocumentChunk]:
        """
        Return the top-n chunks reranked by relevance to the query.
        """
        ...


# ======================================================================
# Gap Detection Interface
# ======================================================================


@runtime_checkable
class GapDetector(Protocol):
    """
    Interface for identifying candidate research gaps from retrieved evidence.

    Implementations may use LLM prompting, rule-based extraction, or hybrid
    approaches.  The detector returns unvalidated CandidateGap objects.
    """

    def detect(
        self,
        query: ResearchQuery,
        chunks: list[DocumentChunk],
    ) -> list[CandidateGap]:
        """
        Detect candidate gaps from a set of retrieved evidence chunks.

        Returns
        -------
        list[CandidateGap]:
            Unranked list of candidate gaps.  Ranking is performed separately.
        """
        ...


# ======================================================================
# EWRGV Validation Interfaces
# ======================================================================


@runtime_checkable
class EvidenceClassifier(Protocol):
    """
    Interface for classifying evidence items collected during EWRGV validation.
    """

    def classify(
        self,
        gap: CandidateGap,
        chunk: DocumentChunk,
    ) -> EvidenceType:
        """
        Classify a chunk as SUPPORTING, COUNTER, PARTIAL, etc. relative to a gap.
        """
        ...


@runtime_checkable
class CoverageAnalyzer(Protocol):
    """
    Interface for the multi-dimensional coverage-analysis step in EWRGV.
    """

    def analyze(
        self,
        gap: CandidateGap,
        evidence: list[Evidence],
    ) -> CoverageAssessment:
        """
        Compute a CoverageAssessment for the given gap and its evidence.
        """
        ...


@runtime_checkable
class ConfidenceScorer(Protocol):
    """
    Interface for the EWRGV confidence-assessment step.

    The concrete formula that combines coverage, evidence strength, and
    evidence counts is intentionally not implemented yet.
    """

    def score(
        self,
        gap: CandidateGap,
        evidence: list[Evidence],
        coverage: CoverageAssessment,
    ) -> float:
        """
        Return a confidence score in [0.0, 1.0].
        """
        ...


@runtime_checkable
class ExplanationGenerator(Protocol):
    """
    Interface for generating human-readable evidence-chain explanations.
    """

    def generate(
        self,
        gap: CandidateGap,
        result: ValidationResult,
    ) -> str:
        """
        Produce a natural-language explanation justifying the classification.
        """
        ...


# ======================================================================
# Storage Interfaces
# ======================================================================


@runtime_checkable
class VectorStore(Protocol):
    """
    Interface for vector-based similarity search backends
    (FAISS, Chroma, Qdrant, …).

    Application code uses this interface; concrete adapters live in
    app/storage/vector/.
    """

    def add(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Index the given chunks with their embeddings."""
        ...

    def search(self, query_embedding: list[float], top_k: int) -> list[DocumentChunk]:
        """Return the top-k most similar chunks."""
        ...

    def delete(self, chunk_ids: list[str]) -> None:
        """Remove chunks by ID."""
        ...


class PaperRepository(ABC):
    """
    Abstract base for persistent Paper storage.

    Using ABC (not Protocol) because concrete implementations will need
    to manage connections and transactions, and we want to enforce the
    interface through inheritance for these stateful repositories.
    """

    @abstractmethod
    def save(self, paper: Paper) -> None: ...

    @abstractmethod
    def get_by_id(self, paper_id: str) -> Optional[Paper]: ...

    @abstractmethod
    def list_all(self) -> list[Paper]: ...

    @abstractmethod
    def delete(self, paper_id: str) -> None: ...


class EvidenceRepository(ABC):
    """Abstract base for persistent Evidence storage."""

    @abstractmethod
    def save(self, evidence: Evidence) -> None: ...

    @abstractmethod
    def get_by_gap_id(self, gap_id: str) -> list[Evidence]: ...

    @abstractmethod
    def get_by_id(self, evidence_id: str) -> Optional[Evidence]: ...

    @abstractmethod
    def delete(self, evidence_id: str) -> None: ...
