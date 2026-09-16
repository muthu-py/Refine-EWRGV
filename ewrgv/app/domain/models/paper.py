"""
app/domain/models/paper.py
--------------------------
Domain models for academic papers and their processed document chunks.

These models represent the *core literature corpus* entities in EWRGV.
They carry only the information that is meaningful at the domain level;
storage-specific types (ORM columns, vector embeddings, etc.) must NOT
appear here.
"""

from __future__ import annotations

from datetime import date
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Author(BaseModel):
    """A single paper author."""

    name: str
    affiliation: Optional[str] = None


class Paper(BaseModel):
    """
    Represents a single academic paper in the EWRGV literature corpus.

    Attributes
    ----------
    paper_id:
        Unique identifier (may be a DOI, Semantic Scholar ID, or generated UUID).
    title:
        Full paper title.
    authors:
        Ordered list of authors.
    abstract:
        Full abstract text.
    publication_date:
        Publication date when available.
    source:
        The provider/database from which the paper was collected
        (e.g. "semantic_scholar", "openalex").
    venue:
        Journal or conference name.
    sections:
        Parsed body sections; keyed by section heading.
    metadata:
        Additional bibliographic or provider-specific metadata.

    Literature-collection fields (Phase 2.2)
    -----------------------------------------
    doi:
        Digital Object Identifier, if available.
    provider_id:
        The paper's native ID from the collection provider
        (e.g. Semantic Scholar paperId, OpenAlex work ID).
    paper_url:
        Canonical URL for the paper landing page.
    full_text_url:
        URL to the full text / PDF when available (Open Access).
    open_access_status:
        OA status string as reported by the provider
        (e.g. "gold", "green", "bronze", "closed").
    citation_count:
        Number of citations as reported by the provider.
    publication_year:
        Four-digit publication year; preferred over publication_date
        because most providers only supply the year.
    """

    paper_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    authors: list[Author] = Field(default_factory=list)
    abstract: str = ""
    publication_date: Optional[date] = None
    source: str = ""
    venue: Optional[str] = None
    sections: dict[str, str] = Field(default_factory=dict)
    metadata: dict = Field(default_factory=dict)

    # ---- Literature-collection fields (Phase 2.2) ---- #
    doi: Optional[str] = None
    provider_id: Optional[str] = None
    paper_url: Optional[str] = None
    full_text_url: Optional[str] = None
    open_access_status: Optional[str] = None
    citation_count: Optional[int] = None
    publication_year: Optional[int] = None


class DocumentChunk(BaseModel):
    """
    A processed text chunk derived from a Paper.

    Papers are split into chunks during the document-processing stage.
    Each chunk is the atomic unit that is embedded, indexed, and retrieved.

    Attributes
    ----------
    chunk_id:
        Unique identifier for this chunk.
    paper_id:
        Reference back to the parent Paper.
    section:
        The section heading this chunk was extracted from.
    text:
        The actual text content of the chunk.
    metadata:
        Chunk-level metadata (e.g. chunk index, character offsets).
    """

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    paper_id: str
    section: str = ""
    text: str
    metadata: dict = Field(default_factory=dict)
