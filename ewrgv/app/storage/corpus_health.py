"""
app/storage/corpus_health.py
------------------------------
Corpus health and consistency checking for Phase 2.8.

Purpose
-------
Detects situations where the database processing status and the actual
state of artifacts in Supabase Storage are out of sync, for example:

  Case 1: DB says parsing_status = 'PARSED'  → parsed.json does NOT exist
  Case 2: DB says chunk_status   = 'CHUNKED' → chunks.json does NOT exist
  Case 3: parsed.json exists                  → DB says parsing_status NOT_PARSED
  Case 4: DB says full_text_status = 'DOWNLOADED' → fulltext.pdf does NOT exist

Inconsistencies are REPORTED, not automatically repaired.

Also provides corpus health summary counts (total / downloaded / parsed /
sections-extracted / chunked), optionally scoped to a single research run.

Design notes
------------
* Uses DocumentStorage.exists() for artifact checks — never downloads content.
* All DB queries go through PaperRepository.
* Inconsistency detection processes papers concurrently (but throttled) to
  avoid hitting storage rate limits.
* Secrets are never logged.
"""

from __future__ import annotations

__all__ = [
    "InconsistencyKind",
    "ArtifactInconsistency",
    "CorpusHealthReport",
    "CorpusHealthService",
]

import asyncio
from enum import Enum
from typing import Optional

from pydantic import BaseModel

from app.core.logging import get_logger
from app.storage.artifact_paths import ArtifactType
from app.storage.document_storage import DocumentStorage

logger = get_logger(__name__)

# Maximum concurrent storage existence checks per health run.
# Keeps this from hammering Supabase Storage rate limits.
_CONCURRENCY_LIMIT = 10


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class InconsistencyKind(str, Enum):
    """Classification of a detected DB / artifact inconsistency."""

    # DB says artifact was produced, but the file is absent in storage
    STATUS_SAYS_DONE_BUT_ARTIFACT_MISSING = "STATUS_SAYS_DONE_BUT_ARTIFACT_MISSING"

    # Artifact exists in storage, but DB status still says NOT_PROCESSED
    ARTIFACT_EXISTS_BUT_STATUS_NOT_UPDATED = "ARTIFACT_EXISTS_BUT_STATUS_NOT_UPDATED"


class ArtifactInconsistency(BaseModel):
    """A single detected inconsistency between DB metadata and storage state."""

    paper_id: str
    artifact_type: ArtifactType
    db_status_field: str       # e.g. "parsing_status"
    db_status_value: str       # e.g. "PARSED"
    artifact_path: str         # e.g. "papers/{id}/parsed.json"
    artifact_exists: bool      # True/False from storage check
    kind: InconsistencyKind
    description: str           # Human-readable explanation


class CorpusHealthSummary(BaseModel):
    """Aggregate processing-status counts for the corpus."""

    run_id: Optional[str] = None     # None means corpus-wide
    total_papers: int = 0
    downloaded: int = 0              # full_text_status = 'DOWNLOADED'
    not_checked: int = 0             # full_text_status = 'NOT_CHECKED'
    no_oa_found: int = 0             # full_text_status = 'NO_OA_FOUND'
    download_failed: int = 0         # full_text_status = 'DOWNLOAD_FAILED'
    invalid_document: int = 0        # full_text_status = 'INVALID_DOCUMENT'
    parsed: int = 0                  # parsing_status = 'PARSED'
    sections_extracted: int = 0      # section_status = 'EXTRACTED'
    chunked: int = 0                 # chunk_status = 'CHUNKED'


class CorpusHealthReport(BaseModel):
    """Full corpus health report: summary + detected inconsistencies."""

    summary: CorpusHealthSummary
    inconsistencies: list[ArtifactInconsistency] = []
    inconsistency_count: int = 0
    papers_checked: int = 0

    def has_inconsistencies(self) -> bool:
        return self.inconsistency_count > 0


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class CorpusHealthService:
    """
    Checks corpus health by cross-referencing DB metadata with storage state.

    Parameters
    ----------
    repo:
        PaperRepository providing DB queries.
    storage:
        DocumentStorage implementation providing existence checks.
    """

    def __init__(self, repo: "PaperRepository", storage: DocumentStorage) -> None:  # type: ignore[name-defined]
        self._repo = repo
        self._storage = storage

    async def get_summary(self, run_id: Optional[str] = None) -> CorpusHealthSummary:
        """
        Return aggregate processing-status counts.

        Parameters
        ----------
        run_id:
            If provided, restrict counts to papers in that research run.
            If None, counts span the entire corpus.
        """
        counts = await self._repo.get_corpus_health_counts(run_id=run_id)
        return CorpusHealthSummary(
            run_id=run_id,
            total_papers=counts.get("total_papers", 0),
            downloaded=counts.get("downloaded", 0),
            not_checked=counts.get("not_checked", 0),
            no_oa_found=counts.get("no_oa_found", 0),
            download_failed=counts.get("download_failed", 0),
            invalid_document=counts.get("invalid_document", 0),
            parsed=counts.get("parsed", 0),
            sections_extracted=counts.get("sections_extracted", 0),
            chunked=counts.get("chunked", 0),
        )

    async def check_consistency(
        self,
        run_id: Optional[str] = None,
        concurrency: int = _CONCURRENCY_LIMIT,
    ) -> list[ArtifactInconsistency]:
        """
        Detect DB / artifact inconsistencies for a run (or the whole corpus).

        For each paper, checks that DB status values align with actual artifact
        existence in storage.  Does NOT repair — only reports.

        Parameters
        ----------
        run_id:
            If provided, restrict checks to papers in that research run.
        concurrency:
            Max concurrent storage existence checks.

        Returns
        -------
        list[ArtifactInconsistency]:
            All detected inconsistencies.  Empty list = healthy.
        """
        papers = await self._repo.get_papers_for_consistency_check(run_id=run_id)
        logger.info(
            "CorpusHealthService.check_consistency starting",
            extra={"paper_count": len(papers), "run_id": run_id},
        )

        semaphore = asyncio.Semaphore(concurrency)
        tasks = [
            self._check_paper(paper=p, semaphore=semaphore)
            for p in papers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        inconsistencies: list[ArtifactInconsistency] = []
        for res in results:
            if isinstance(res, Exception):
                logger.warning(
                    "CorpusHealthService: error checking paper consistency",
                    extra={"error": str(res)},
                )
            elif isinstance(res, list):
                inconsistencies.extend(res)

        return inconsistencies

    async def get_full_report(
        self,
        run_id: Optional[str] = None,
        check_artifacts: bool = True,
    ) -> CorpusHealthReport:
        """
        Return a full health report: summary + (optionally) inconsistencies.

        Parameters
        ----------
        run_id:
            Scope to a single research run, or None for corpus-wide.
        check_artifacts:
            If True (default), perform storage existence checks and report
            inconsistencies.  If False, return summary only (faster).
        """
        summary = await self.get_summary(run_id=run_id)

        inconsistencies: list[ArtifactInconsistency] = []
        papers_checked = 0

        if check_artifacts:
            papers = await self._repo.get_papers_for_consistency_check(run_id=run_id)
            papers_checked = len(papers)
            inconsistencies = await self.check_consistency(run_id=run_id)

        return CorpusHealthReport(
            summary=summary,
            inconsistencies=inconsistencies,
            inconsistency_count=len(inconsistencies),
            papers_checked=papers_checked,
        )

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    async def _check_paper(
        self,
        paper: dict,
        semaphore: asyncio.Semaphore,
    ) -> list[ArtifactInconsistency]:
        """
        Check one paper for DB/storage inconsistencies.

        Returns a list of ArtifactInconsistency objects (possibly empty).
        """
        async with semaphore:
            paper_id = paper["paper_id"]
            inconsistencies: list[ArtifactInconsistency] = []

            checks = _build_consistency_checks(paper)
            for check in checks:
                try:
                    exists = await self._storage.exists(paper_id, check["artifact_type"])
                    artifact_path = self._storage.get_path(paper_id, check["artifact_type"])

                    if check["expect_exists"] and not exists:
                        inconsistencies.append(
                            ArtifactInconsistency(
                                paper_id=paper_id,
                                artifact_type=check["artifact_type"],
                                db_status_field=check["status_field"],
                                db_status_value=check["status_value"],
                                artifact_path=artifact_path,
                                artifact_exists=False,
                                kind=InconsistencyKind.STATUS_SAYS_DONE_BUT_ARTIFACT_MISSING,
                                description=(
                                    f"DB {check['status_field']}={check['status_value']} "
                                    f"implies artifact should exist, "
                                    f"but '{artifact_path}' was not found in storage."
                                ),
                            )
                        )
                    elif check["expect_missing"] and exists:
                        inconsistencies.append(
                            ArtifactInconsistency(
                                paper_id=paper_id,
                                artifact_type=check["artifact_type"],
                                db_status_field=check["status_field"],
                                db_status_value=check["status_value"],
                                artifact_path=artifact_path,
                                artifact_exists=True,
                                kind=InconsistencyKind.ARTIFACT_EXISTS_BUT_STATUS_NOT_UPDATED,
                                description=(
                                    f"Artifact '{artifact_path}' exists in storage, "
                                    f"but DB {check['status_field']}={check['status_value']} "
                                    f"suggests it was not processed."
                                ),
                            )
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "CorpusHealthService: storage existence check error",
                        extra={
                            "paper_id": paper_id,
                            "artifact_type": check["artifact_type"],
                            "error": str(exc),
                        },
                    )

            return inconsistencies


# ---------------------------------------------------------------------------
# Consistency rule builder
# ---------------------------------------------------------------------------


def _build_consistency_checks(paper: dict) -> list[dict]:
    """
    Build a list of consistency checks for a single paper row.

    Each check specifies:
      - artifact_type: which artifact to verify
      - status_field:  which DB column was inspected
      - status_value:  the value found in that column
      - expect_exists: True if the artifact SHOULD be present
      - expect_missing: True if the artifact SHOULD be absent

    Rules implemented:
      1. full_text_status = DOWNLOADED   → fulltext.pdf should exist
      2. parsing_status   = PARSED       → parsed.json should exist
      3. section_status   = EXTRACTED    → sections.json should exist
      4. chunk_status     = CHUNKED      → chunks.json should exist
      5. parsing_status   = NOT_PARSED   → parsed.json likely should not exist
         (Case 3 from spec: artifact exists but status says NOT_PARSED)
    """
    checks: list[dict] = []

    full_text_status = paper.get("full_text_status") or "NOT_CHECKED"
    parsing_status = paper.get("parsing_status") or "NOT_PARSED"
    section_status = paper.get("section_status") or "NOT_EXTRACTED"
    chunk_status = paper.get("chunk_status") or "NOT_CHUNKED"

    # Rule 1: DOWNLOADED → fulltext.pdf should exist
    if full_text_status == "DOWNLOADED":
        checks.append({
            "artifact_type": ArtifactType.FULL_TEXT,
            "status_field": "full_text_status",
            "status_value": full_text_status,
            "expect_exists": True,
            "expect_missing": False,
        })

    # Rule 2: PARSED → parsed.json should exist
    if parsing_status == "PARSED":
        checks.append({
            "artifact_type": ArtifactType.PARSED_DOCUMENT,
            "status_field": "parsing_status",
            "status_value": parsing_status,
            "expect_exists": True,
            "expect_missing": False,
        })

    # Rule 3 (inverse): NOT_PARSED but parsed.json present
    if parsing_status == "NOT_PARSED":
        checks.append({
            "artifact_type": ArtifactType.PARSED_DOCUMENT,
            "status_field": "parsing_status",
            "status_value": parsing_status,
            "expect_exists": False,
            "expect_missing": True,
        })

    # Rule 4: EXTRACTED → sections.json should exist
    if section_status == "EXTRACTED":
        checks.append({
            "artifact_type": ArtifactType.SECTIONS,
            "status_field": "section_status",
            "status_value": section_status,
            "expect_exists": True,
            "expect_missing": False,
        })

    # Rule 5: CHUNKED → chunks.json should exist
    if chunk_status == "CHUNKED":
        checks.append({
            "artifact_type": ArtifactType.CHUNKS,
            "status_field": "chunk_status",
            "status_value": chunk_status,
            "expect_exists": True,
            "expect_missing": False,
        })

    return checks
