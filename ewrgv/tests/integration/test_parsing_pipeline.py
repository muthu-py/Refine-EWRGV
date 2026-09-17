"""
tests/integration/test_parsing_pipeline.py
-------------------------------------------
Integration test for the Phase 2.5 document parsing pipeline.

Verifies the end-to-end conceptual flow:

    DOWNLOADED paper
        |
        v
    PDFParser.parse()
        |
        v
    ParsedDocument
        |
        v
    parsed.json uploaded to Storage
        |
        v
    parsing_status = PARSED in DB

Uses mocked Storage and DB to avoid requiring a live Supabase environment.
"""

from __future__ import annotations

import io
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pymupdf

from app.core.exceptions import StorageError
from app.core.job_store import JobStatus, JobStore
from app.domain.models.parsed_document import ParsedDocument
from app.ingestion.parsers.pdf_parser import PDFParser
from app.ingestion.parsers.worker import run_parsing
from app.storage.supabase_storage import SupabaseStorageClient


# ------------------------------------------------------------------ #
# Fixture: synthetic PDF bytes
# ------------------------------------------------------------------ #

def make_realistic_pdf(num_pages: int = 4, chars_per_page: int = 500) -> bytes:
    """Create a multi-page PDF with meaningful text content."""
    doc = pymupdf.open()
    for i in range(num_pages):
        page = doc.new_page()
        text = f"Page {i + 1}. " + ("Lorem ipsum dolor sit amet, " * (chars_per_page // 30))
        rect = pymupdf.Rect(50, 50, 500, 750)
        page.insert_textbox(rect, text, fontsize=10)
    doc.set_metadata({"title": "Integration Test Paper", "creator": "pytest"})
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


# ------------------------------------------------------------------ #
# Integration: PDFParser directly (no worker)
# ------------------------------------------------------------------ #

class TestPDFParserIntegration:
    @pytest.mark.asyncio
    async def test_parse_realistic_pdf(self):
        """Parse a realistic 4-page PDF and verify the ParsedDocument structure."""
        pdf_bytes = make_realistic_pdf(num_pages=4)

        mock_storage = MagicMock()
        mock_storage.download = AsyncMock(return_value=pdf_bytes)
        mock_storage.close = AsyncMock()

        parser = PDFParser(storage_client=mock_storage, bucket="papers")
        doc = await parser.parse("integration-paper-1", "papers/integration-paper-1/fulltext.pdf")

        # Verify structure
        assert isinstance(doc, ParsedDocument)
        assert doc.paper_id == "integration-paper-1"
        assert doc.page_count == 4
        assert len(doc.pages) == 4

        # Verify 1-based page numbering
        for i, page in enumerate(doc.pages):
            assert page.page_number == i + 1

        # Verify meaningful text was extracted
        assert doc.total_character_count > 100
        assert doc.pages_with_text > 0

        # Verify each page has text
        for page in doc.pages:
            assert f"Page {page.page_number}" in page.text

        # Verify metadata captured
        assert isinstance(doc.metadata, dict)

    @pytest.mark.asyncio
    async def test_parse_to_json_is_valid(self):
        """Verify the ParsedDocument serialises to valid JSON."""
        pdf_bytes = make_realistic_pdf(num_pages=2)

        mock_storage = MagicMock()
        mock_storage.download = AsyncMock(return_value=pdf_bytes)

        parser = PDFParser(storage_client=mock_storage, bucket="papers")
        doc = await parser.parse("paper-json-test", "papers/paper-json-test/fulltext.pdf")

        json_str = doc.model_dump_json(indent=2)
        data = json.loads(json_str)

        assert data["paper_id"] == "paper-json-test"
        assert len(data["pages"]) == 2
        assert data["page_count"] == 2
        assert all("page_number" in p for p in data["pages"])
        assert all("text" in p for p in data["pages"])
        assert all("character_count" in p for p in data["pages"])


# ------------------------------------------------------------------ #
# Integration: Full worker pipeline with mocked repo and storage
# ------------------------------------------------------------------ #

class TestParsingWorkerIntegration:
    @pytest.mark.asyncio
    async def test_full_pipeline_downloaded_to_parsed(self):
        """
        End-to-end: DOWNLOADED paper -> worker -> ParsedDocument
        -> parsed.json uploaded -> parsing_status = PARSED.
        """
        pdf_bytes = make_realistic_pdf(num_pages=3)

        # Track what was uploaded to "storage"
        uploaded: dict[str, bytes] = {}
        updated: list[dict] = []

        async def mock_upload(bucket, path, content, **kwargs):
            uploaded[path] = content
            return path

        async def mock_update(paper_id, status, **kwargs):
            updated.append({"paper_id": paper_id, "status": status, **kwargs})

        papers = [{"paper_id": "worker-paper-1", "storage_path": "papers/worker-paper-1/fulltext.pdf"}]

        mock_repo = MagicMock()
        mock_repo.get_papers_for_parsing = AsyncMock(return_value=papers)
        mock_repo.update_parsing_metadata = AsyncMock(side_effect=mock_update)

        with patch("app.ingestion.parsers.worker.SupabaseStorageClient") as mock_sc_cls:
            mock_sc = MagicMock()
            mock_sc.download = AsyncMock(return_value=pdf_bytes)
            mock_sc.upload = AsyncMock(side_effect=mock_upload)
            mock_sc.close = AsyncMock()
            mock_sc_cls.return_value = mock_sc

            job_store = JobStore()
            job = await job_store.create_parsing_job("integration-run-1")
            await run_parsing(job=job, job_store=job_store, repo=mock_repo)

        # Job must complete successfully
        assert job.status == JobStatus.COMPLETED
        assert job.parsed == 1
        assert job.failed == 0

        # parsed.json must have been uploaded
        expected_path = "papers/worker-paper-1/parsed.json"
        assert expected_path in uploaded

        # Uploaded content must be valid JSON with correct structure
        uploaded_data = json.loads(uploaded[expected_path].decode("utf-8"))
        assert uploaded_data["paper_id"] == "worker-paper-1"
        assert uploaded_data["page_count"] == 3
        assert len(uploaded_data["pages"]) == 3

        # DB must have been updated with PARSED status
        assert len(updated) == 1
        assert updated[0]["paper_id"] == "worker-paper-1"
        assert updated[0]["status"] == "PARSED"
        assert updated[0].get("page_count") == 3
        assert updated[0].get("parsed_storage_path") == expected_path

    @pytest.mark.asyncio
    async def test_idempotency_no_papers_to_parse(self):
        """If all papers are already PARSED, the worker completes with 0 processed."""
        mock_repo = MagicMock()
        mock_repo.get_papers_for_parsing = AsyncMock(return_value=[])

        with patch("app.ingestion.parsers.worker.SupabaseStorageClient") as mock_sc_cls:
            mock_sc = MagicMock()
            mock_sc.close = AsyncMock()
            mock_sc_cls.return_value = mock_sc

            job_store = JobStore()
            job = await job_store.create_parsing_job("integration-run-idempotent")
            await run_parsing(job=job, job_store=job_store, repo=mock_repo)

        assert job.status == JobStatus.COMPLETED
        assert job.total_papers == 0
        assert job.parsed == 0

    @pytest.mark.asyncio
    async def test_mixed_success_and_failure(self):
        """Papers B fails, A and C succeed. Worker must complete and report correctly."""
        good_pdf = make_realistic_pdf(num_pages=2)

        papers = [
            {"paper_id": "pa", "storage_path": "papers/pa/fulltext.pdf"},
            {"paper_id": "pb", "storage_path": "papers/pb/fulltext.pdf"},  # will fail
            {"paper_id": "pc", "storage_path": "papers/pc/fulltext.pdf"},
        ]

        failed_ids: list[str] = []
        parsed_ids: list[str] = []

        async def mock_update(paper_id, status, **kwargs):
            if status == "PARSED":
                parsed_ids.append(paper_id)
            else:
                failed_ids.append(paper_id)

        async def mock_download(bucket, path):
            if "pb" in path:
                raise StorageError("Simulated download failure for pb")
            return good_pdf

        mock_repo = MagicMock()
        mock_repo.get_papers_for_parsing = AsyncMock(return_value=papers)
        mock_repo.update_parsing_metadata = AsyncMock(side_effect=mock_update)

        with patch("app.ingestion.parsers.worker.SupabaseStorageClient") as mock_sc_cls:
            mock_sc = MagicMock()
            mock_sc.download = AsyncMock(side_effect=mock_download)
            mock_sc.upload = AsyncMock(return_value="path")
            mock_sc.close = AsyncMock()
            mock_sc_cls.return_value = mock_sc

            job_store = JobStore()
            job = await job_store.create_parsing_job("integration-run-mixed")
            await run_parsing(job=job, job_store=job_store, repo=mock_repo)

        assert job.status == JobStatus.COMPLETED
        assert job.parsed == 2   # pa and pc
        assert job.failed == 1   # pb
        assert "pa" in parsed_ids
        assert "pc" in parsed_ids
        assert "pb" in failed_ids
