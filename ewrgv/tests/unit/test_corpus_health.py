"""
tests/unit/test_corpus_health.py
-----------------------------------
Unit tests for app/storage/corpus_health.py (Phase 2.8).

Tests verify:
- Health summary counts from DB
- Inconsistency detection (all 4 cases from spec)
- Provider failure isolation
- Orphan artifact detection (artifact exists, DB says not processed)
- Clean paper produces no inconsistencies
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.storage.artifact_paths import ArtifactType
from app.storage.corpus_health import (
    CorpusHealthService,
    InconsistencyKind,
    _build_consistency_checks,
)


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #


def _make_mock_repo(papers: list[dict], counts: dict | None = None):
    """Build a mock PaperRepository."""
    repo = MagicMock()
    repo.get_papers_for_consistency_check = AsyncMock(return_value=papers)
    repo.get_corpus_health_counts = AsyncMock(return_value=counts or {
        "total_papers": len(papers),
        "downloaded": 0,
        "not_checked": 0,
        "no_oa_found": 0,
        "download_failed": 0,
        "invalid_document": 0,
        "parsed": 0,
        "sections_extracted": 0,
        "chunked": 0,
    })
    return repo


def _make_mock_storage(exists_map: dict[tuple[str, ArtifactType], bool]):
    """
    Build a mock DocumentStorage where exists() returns a value per (paper_id, artifact_type).
    """
    storage = MagicMock()

    async def mock_exists(paper_id, artifact_type):
        return exists_map.get((paper_id, artifact_type), False)

    storage.exists = mock_exists
    storage.get_path = MagicMock(
        side_effect=lambda pid, at: f"papers/{pid}/{at.value.lower()}"
    )
    return storage


# ------------------------------------------------------------------ #
# Health summary counts
# ------------------------------------------------------------------ #


class TestHealthSummary:
    @pytest.mark.asyncio
    async def test_health_summary_correct_counts(self):
        expected_counts = {
            "total_papers": 50,
            "downloaded": 23,
            "not_checked": 20,
            "no_oa_found": 5,
            "download_failed": 2,
            "invalid_document": 0,
            "parsed": 20,
            "sections_extracted": 20,
            "chunked": 18,
        }
        repo = _make_mock_repo(papers=[], counts=expected_counts)
        storage = _make_mock_storage({})
        service = CorpusHealthService(repo=repo, storage=storage)

        summary = await service.get_summary(run_id="run-1")

        assert summary.total_papers == 50
        assert summary.downloaded == 23
        assert summary.parsed == 20
        assert summary.sections_extracted == 20
        assert summary.chunked == 18

    @pytest.mark.asyncio
    async def test_health_summary_run_id_forwarded(self):
        repo = _make_mock_repo(papers=[])
        storage = _make_mock_storage({})
        service = CorpusHealthService(repo=repo, storage=storage)

        await service.get_summary(run_id="specific-run")
        repo.get_corpus_health_counts.assert_awaited_once_with(run_id="specific-run")

    @pytest.mark.asyncio
    async def test_health_summary_corpus_wide(self):
        repo = _make_mock_repo(papers=[])
        storage = _make_mock_storage({})
        service = CorpusHealthService(repo=repo, storage=storage)

        await service.get_summary(run_id=None)
        repo.get_corpus_health_counts.assert_awaited_once_with(run_id=None)


# ------------------------------------------------------------------ #
# Consistency checks — _build_consistency_checks
# ------------------------------------------------------------------ #


class TestBuildConsistencyChecks:
    """Unit tests for the rule builder without any async I/O."""

    def _paper(self, **kwargs):
        defaults = {
            "paper_id": "p1",
            "full_text_status": "NOT_CHECKED",
            "parsing_status": "NOT_PARSED",
            "section_status": "NOT_EXTRACTED",
            "chunk_status": "NOT_CHUNKED",
        }
        defaults.update(kwargs)
        return defaults

    def test_downloaded_expects_fulltext_exists(self):
        paper = self._paper(full_text_status="DOWNLOADED")
        checks = _build_consistency_checks(paper)
        fulltext_checks = [c for c in checks if c["artifact_type"] == ArtifactType.FULL_TEXT]
        assert any(c["expect_exists"] for c in fulltext_checks)

    def test_parsed_expects_parsed_json_exists(self):
        paper = self._paper(parsing_status="PARSED")
        checks = _build_consistency_checks(paper)
        parsed_checks = [c for c in checks if c["artifact_type"] == ArtifactType.PARSED_DOCUMENT]
        assert any(c["expect_exists"] for c in parsed_checks)

    def test_not_parsed_expects_parsed_json_missing(self):
        """Case 3 from spec: artifact exists but status says NOT_PARSED."""
        paper = self._paper(parsing_status="NOT_PARSED")
        checks = _build_consistency_checks(paper)
        parsed_checks = [c for c in checks if c["artifact_type"] == ArtifactType.PARSED_DOCUMENT]
        assert any(c["expect_missing"] for c in parsed_checks)

    def test_extracted_expects_sections_json_exists(self):
        paper = self._paper(section_status="EXTRACTED")
        checks = _build_consistency_checks(paper)
        section_checks = [c for c in checks if c["artifact_type"] == ArtifactType.SECTIONS]
        assert any(c["expect_exists"] for c in section_checks)

    def test_chunked_expects_chunks_json_exists(self):
        paper = self._paper(chunk_status="CHUNKED")
        checks = _build_consistency_checks(paper)
        chunk_checks = [c for c in checks if c["artifact_type"] == ArtifactType.CHUNKS]
        assert any(c["expect_exists"] for c in chunk_checks)

    def test_fully_processed_paper_has_four_checks(self):
        """A fully processed paper should have checks for all 4 artifact types."""
        paper = self._paper(
            full_text_status="DOWNLOADED",
            parsing_status="PARSED",
            section_status="EXTRACTED",
            chunk_status="CHUNKED",
        )
        checks = _build_consistency_checks(paper)
        types_checked = {c["artifact_type"] for c in checks}
        assert ArtifactType.FULL_TEXT in types_checked
        assert ArtifactType.PARSED_DOCUMENT in types_checked
        assert ArtifactType.SECTIONS in types_checked
        assert ArtifactType.CHUNKS in types_checked

    def test_not_checked_paper_has_minimal_checks(self):
        """A NOT_CHECKED paper has only the 'NOT_PARSED implies parsed.json missing' check."""
        paper = self._paper()
        checks = _build_consistency_checks(paper)
        # Only the NOT_PARSED → expect_missing check
        assert all(c["expect_missing"] for c in checks)


# ------------------------------------------------------------------ #
# Case 1: DB says PARSED, parsed.json MISSING → inconsistency
# ------------------------------------------------------------------ #


class TestCase1StatusDoneArtifactMissing:
    @pytest.mark.asyncio
    async def test_parsed_status_but_missing_artifact(self):
        """
        Case 1: DB says parsing_status = PARSED
                but parsed.json does NOT exist → inconsistency
        """
        paper_id = "paper-case1"
        papers = [{
            "paper_id": paper_id,
            "full_text_status": "DOWNLOADED",
            "parsing_status": "PARSED",
            "section_status": "NOT_EXTRACTED",
            "chunk_status": "NOT_CHUNKED",
        }]
        exists_map = {
            (paper_id, ArtifactType.FULL_TEXT): True,
            (paper_id, ArtifactType.PARSED_DOCUMENT): False,  # missing!
        }
        repo = _make_mock_repo(papers=papers)
        storage = _make_mock_storage(exists_map)
        service = CorpusHealthService(repo=repo, storage=storage)

        inconsistencies = await service.check_consistency()

        assert len(inconsistencies) == 1
        inc = inconsistencies[0]
        assert inc.paper_id == paper_id
        assert inc.artifact_type == ArtifactType.PARSED_DOCUMENT
        assert inc.kind == InconsistencyKind.STATUS_SAYS_DONE_BUT_ARTIFACT_MISSING
        assert inc.artifact_exists is False

    @pytest.mark.asyncio
    async def test_chunked_status_but_missing_artifact(self):
        """Case 2: DB says chunk_status = CHUNKED but chunks.json missing."""
        paper_id = "paper-case2"
        papers = [{
            "paper_id": paper_id,
            "full_text_status": "DOWNLOADED",
            "parsing_status": "PARSED",
            "section_status": "EXTRACTED",
            "chunk_status": "CHUNKED",
        }]
        exists_map = {
            (paper_id, ArtifactType.FULL_TEXT): True,
            (paper_id, ArtifactType.PARSED_DOCUMENT): True,
            (paper_id, ArtifactType.SECTIONS): True,
            (paper_id, ArtifactType.CHUNKS): False,  # missing!
        }
        repo = _make_mock_repo(papers=papers)
        storage = _make_mock_storage(exists_map)
        service = CorpusHealthService(repo=repo, storage=storage)

        inconsistencies = await service.check_consistency()

        chunk_incs = [i for i in inconsistencies if i.artifact_type == ArtifactType.CHUNKS]
        assert len(chunk_incs) == 1
        assert chunk_incs[0].kind == InconsistencyKind.STATUS_SAYS_DONE_BUT_ARTIFACT_MISSING

    @pytest.mark.asyncio
    async def test_downloaded_status_but_missing_pdf(self):
        """Case 4: DB says full_text_status = DOWNLOADED but fulltext.pdf missing."""
        paper_id = "paper-case4"
        papers = [{
            "paper_id": paper_id,
            "full_text_status": "DOWNLOADED",
            "parsing_status": "NOT_PARSED",
            "section_status": "NOT_EXTRACTED",
            "chunk_status": "NOT_CHUNKED",
        }]
        exists_map = {
            (paper_id, ArtifactType.FULL_TEXT): False,  # missing!
        }
        repo = _make_mock_repo(papers=papers)
        storage = _make_mock_storage(exists_map)
        service = CorpusHealthService(repo=repo, storage=storage)

        inconsistencies = await service.check_consistency()

        pdf_incs = [i for i in inconsistencies if i.artifact_type == ArtifactType.FULL_TEXT]
        assert len(pdf_incs) == 1
        assert pdf_incs[0].kind == InconsistencyKind.STATUS_SAYS_DONE_BUT_ARTIFACT_MISSING


# ------------------------------------------------------------------ #
# Case 3: Artifact exists, DB status says NOT processed → inconsistency
# ------------------------------------------------------------------ #


class TestCase3ArtifactExistsStatusNotUpdated:
    @pytest.mark.asyncio
    async def test_parsed_json_exists_but_db_says_not_parsed(self):
        """Case 3: parsed.json exists but parsing_status = NOT_PARSED → inconsistency."""
        paper_id = "paper-case3"
        papers = [{
            "paper_id": paper_id,
            "full_text_status": "DOWNLOADED",
            "parsing_status": "NOT_PARSED",
            "section_status": "NOT_EXTRACTED",
            "chunk_status": "NOT_CHUNKED",
        }]
        exists_map = {
            (paper_id, ArtifactType.FULL_TEXT): True,
            (paper_id, ArtifactType.PARSED_DOCUMENT): True,  # exists but status not updated
        }
        repo = _make_mock_repo(papers=papers)
        storage = _make_mock_storage(exists_map)
        service = CorpusHealthService(repo=repo, storage=storage)

        inconsistencies = await service.check_consistency()

        parsed_incs = [
            i for i in inconsistencies
            if i.artifact_type == ArtifactType.PARSED_DOCUMENT
        ]
        assert len(parsed_incs) == 1
        assert parsed_incs[0].kind == InconsistencyKind.ARTIFACT_EXISTS_BUT_STATUS_NOT_UPDATED
        assert parsed_incs[0].artifact_exists is True


# ------------------------------------------------------------------ #
# Clean paper — no inconsistencies
# ------------------------------------------------------------------ #


class TestCleanPaper:
    @pytest.mark.asyncio
    async def test_fully_processed_paper_with_all_artifacts_is_clean(self):
        paper_id = "paper-clean"
        papers = [{
            "paper_id": paper_id,
            "full_text_status": "DOWNLOADED",
            "parsing_status": "PARSED",
            "section_status": "EXTRACTED",
            "chunk_status": "CHUNKED",
        }]
        exists_map = {
            (paper_id, ArtifactType.FULL_TEXT): True,
            (paper_id, ArtifactType.PARSED_DOCUMENT): True,
            (paper_id, ArtifactType.SECTIONS): True,
            (paper_id, ArtifactType.CHUNKS): True,
        }
        repo = _make_mock_repo(papers=papers)
        storage = _make_mock_storage(exists_map)
        service = CorpusHealthService(repo=repo, storage=storage)

        inconsistencies = await service.check_consistency()
        assert inconsistencies == []


# ------------------------------------------------------------------ #
# Failure isolation
# ------------------------------------------------------------------ #


class TestFailureIsolation:
    @pytest.mark.asyncio
    async def test_one_paper_error_does_not_stop_others(self):
        """
        A storage existence check error for one paper must not prevent
        other papers from being checked.
        """
        paper_a = "paper-A"
        paper_b = "paper-B"
        papers = [
            {
                "paper_id": paper_a,
                "full_text_status": "DOWNLOADED",
                "parsing_status": "PARSED",
                "section_status": "NOT_EXTRACTED",
                "chunk_status": "NOT_CHUNKED",
            },
            {
                "paper_id": paper_b,
                "full_text_status": "DOWNLOADED",
                "parsing_status": "PARSED",
                "section_status": "NOT_EXTRACTED",
                "chunk_status": "NOT_CHUNKED",
            },
        ]

        call_count = {"n": 0}

        async def flaky_exists(paper_id, artifact_type):
            call_count["n"] += 1
            if paper_id == paper_a and artifact_type == ArtifactType.PARSED_DOCUMENT:
                raise Exception("Network error for paper A")
            # Paper B has parsed.json missing → inconsistency
            return artifact_type == ArtifactType.FULL_TEXT

        storage = MagicMock()
        storage.exists = flaky_exists
        storage.get_path = MagicMock(
            side_effect=lambda pid, at: f"papers/{pid}/{at.value.lower()}"
        )

        repo = _make_mock_repo(papers=papers)
        service = CorpusHealthService(repo=repo, storage=storage)

        # Should not raise; paper A's error is swallowed, paper B is still checked
        inconsistencies = await service.check_consistency()

        # Paper B should have a parsed inconsistency
        paper_b_incs = [i for i in inconsistencies if i.paper_id == paper_b]
        assert len(paper_b_incs) >= 1

    @pytest.mark.asyncio
    async def test_no_papers_returns_empty_inconsistencies(self):
        repo = _make_mock_repo(papers=[])
        storage = _make_mock_storage({})
        service = CorpusHealthService(repo=repo, storage=storage)

        inconsistencies = await service.check_consistency()
        assert inconsistencies == []


# ------------------------------------------------------------------ #
# Full report
# ------------------------------------------------------------------ #


class TestFullReport:
    @pytest.mark.asyncio
    async def test_full_report_structure(self):
        counts = {
            "total_papers": 10,
            "downloaded": 5,
            "not_checked": 5,
            "no_oa_found": 0,
            "download_failed": 0,
            "invalid_document": 0,
            "parsed": 3,
            "sections_extracted": 2,
            "chunked": 1,
        }
        repo = _make_mock_repo(papers=[], counts=counts)
        storage = _make_mock_storage({})
        service = CorpusHealthService(repo=repo, storage=storage)

        report = await service.get_full_report(run_id="run-x", check_artifacts=False)

        assert report.summary.total_papers == 10
        assert report.summary.parsed == 3
        assert report.inconsistency_count == 0
        assert report.has_inconsistencies() is False
