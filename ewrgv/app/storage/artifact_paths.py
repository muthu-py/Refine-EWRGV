"""
app/storage/artifact_paths.py
------------------------------
Centralized, deterministic artifact path construction for EWRGV corpus storage.

Design notes
------------
* All four artifact types share a single canonical hierarchy under the
  Supabase Storage "papers" bucket:

    papers/{paper_id}/fulltext.pdf
    papers/{paper_id}/parsed.json
    papers/{paper_id}/sections.json
    papers/{paper_id}/chunks.json

* Paths returned here are BUCKET-RELATIVE — they do NOT include the bucket
  name as a prefix.  The bucket name is supplied separately to the storage
  client call.

* NEVER hardcode these paths elsewhere in the codebase.  Import from here.

* Verified against the Supabase Storage REST API:
    POST /storage/v1/object/{bucket}/{path}
  where {bucket} = "papers" and {path} = one of the values below.
  This correctly produces:
    /storage/v1/object/papers/papers/{paper_id}/fulltext.pdf
  with no double-prefix because the path itself starts with "papers/{paper_id}/".

Why "papers/" inside the path?
  The folder-within-bucket organizes objects by paper ID, making per-paper
  directories visible in the Supabase Storage UI.
"""

from __future__ import annotations

__all__ = [
    "ARTIFACT_BUCKET",
    "ArtifactType",
    "get_artifact_path",
    "get_fulltext_path",
    "get_parsed_path",
    "get_sections_path",
    "get_chunks_path",
    "parse_paper_id_from_path",
]

from enum import Enum

from app.core.config import settings


# ---------------------------------------------------------------------------
# Bucket constant
# ---------------------------------------------------------------------------

# Centralised so callers never guess the bucket name.
ARTIFACT_BUCKET: str = settings.SUPABASE_STORAGE_BUCKET  # "papers" by default


# ---------------------------------------------------------------------------
# Artifact types
# ---------------------------------------------------------------------------


class ArtifactType(str, Enum):
    """
    The four artifact types produced during corpus processing.

    Each maps to a deterministic filename within a paper's folder.
    """

    FULL_TEXT = "FULL_TEXT"
    """Original PDF download.  fulltext.pdf"""

    PARSED_DOCUMENT = "PARSED_DOCUMENT"
    """JSON output of the PDF parser.  parsed.json"""

    SECTIONS = "SECTIONS"
    """JSON output of section extraction.  sections.json"""

    CHUNKS = "CHUNKS"
    """JSON output of chunking.  chunks.json"""


# Internal mapping: ArtifactType → filename inside papers/{paper_id}/
_ARTIFACT_FILENAMES: dict[ArtifactType, str] = {
    ArtifactType.FULL_TEXT: "fulltext.pdf",
    ArtifactType.PARSED_DOCUMENT: "parsed.json",
    ArtifactType.SECTIONS: "sections.json",
    ArtifactType.CHUNKS: "chunks.json",
}


# ---------------------------------------------------------------------------
# Path construction
# ---------------------------------------------------------------------------


def get_artifact_path(paper_id: str, artifact_type: ArtifactType) -> str:
    """
    Return the bucket-relative path for a given paper and artifact type.

    Parameters
    ----------
    paper_id:
        UUID string of the canonical paper (as stored in ``papers.paper_id``).
    artifact_type:
        One of the four :class:`ArtifactType` values.

    Returns
    -------
    str
        Bucket-relative path, e.g. ``papers/abc123.../fulltext.pdf``.
        This path does NOT start with the bucket name.

    Examples
    --------
    >>> get_artifact_path("abc", ArtifactType.FULL_TEXT)
    'papers/abc/fulltext.pdf'
    >>> get_artifact_path("abc", ArtifactType.PARSED_DOCUMENT)
    'papers/abc/parsed.json'
    """
    if not paper_id:
        raise ValueError("paper_id must not be empty")
    filename = _ARTIFACT_FILENAMES[artifact_type]
    return f"papers/{paper_id}/{filename}"


def get_fulltext_path(paper_id: str) -> str:
    """Return bucket-relative path for the raw PDF artifact."""
    return get_artifact_path(paper_id, ArtifactType.FULL_TEXT)


def get_parsed_path(paper_id: str) -> str:
    """Return bucket-relative path for the parsed JSON artifact."""
    return get_artifact_path(paper_id, ArtifactType.PARSED_DOCUMENT)


def get_sections_path(paper_id: str) -> str:
    """Return bucket-relative path for the sections JSON artifact."""
    return get_artifact_path(paper_id, ArtifactType.SECTIONS)


def get_chunks_path(paper_id: str) -> str:
    """Return bucket-relative path for the chunks JSON artifact."""
    return get_artifact_path(paper_id, ArtifactType.CHUNKS)


# ---------------------------------------------------------------------------
# Reverse lookup
# ---------------------------------------------------------------------------


def parse_paper_id_from_path(path: str) -> str | None:
    """
    Extract the paper_id from a bucket-relative artifact path.

    Handles both clean paths (``papers/{id}/filename``) and the legacy
    double-prefix form (``papers/papers/{id}/filename``) produced by a
    bug in earlier Phase 2.4 code.

    Parameters
    ----------
    path:
        A bucket-relative storage path.

    Returns
    -------
    str | None
        The paper_id segment, or None if the path cannot be parsed.

    Examples
    --------
    >>> parse_paper_id_from_path("papers/abc123/fulltext.pdf")
    'abc123'
    >>> parse_paper_id_from_path("papers/papers/abc123/fulltext.pdf")
    'abc123'
    """
    if not path:
        return None

    # Strip accidental bucket prefix (legacy double-prefix bug)
    if path.startswith("papers/papers/"):
        path = path[len("papers/"):]

    parts = path.split("/")
    # Expected: ["papers", "{paper_id}", "{filename}"]
    if len(parts) >= 3 and parts[0] == "papers":
        return parts[1]
    return None
