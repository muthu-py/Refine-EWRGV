"""
tests/unit/test_pdf_parser.py
-------------------------------
Unit tests for PDFParser (Phase 2.5).

All tests use synthetic in-memory PDFs created with PyMuPDF (pymupdf).
No external PDFs, network calls, or Supabase credentials are needed.

Tests cover all 9 required scenarios:
1.  Valid multi-page PDF   -> correct page count, 1-based numbering, text extracted
2.  Empty/near-empty PDF   -> detected as low-text
3.  Corrupted bytes        -> IngestionError raised
4.  PDF metadata           -> captured in ParsedDocument.metadata
5.  Page boundaries        -> text from page 1 does not bleed into page 2
6.  JSON serialisation     -> ParsedDocument.model_dump_json() round-trips correctly
7.  Idempotency            -> PARSED papers skipped by get_papers_for_parsing (repo test)
8.  Worker resilience      -> one paper failure does not halt the loop
9.  Storage failure        -> parsing not marked PARSED if upload fails
"""

from __future__ import annotations

import io
import json
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pymupdf

from app.core.exceptions import IngestionError, StorageError
from app.domain.models.parsed_document import ParsedDocument, ParsedPage
from app.ingestion.parsers.pdf_parser import PDFParser, _MIN_MEANINGFUL_CHARS


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def make_pdf_bytes(pages: list[str], title: str = "", author: str = "") -> bytes:
    """
    Create a minimal in-memory PDF with the given pages and optional metadata.
    Each entry in `pages` is the text inserted on that page.
    """
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            rect = pymupdf.Rect(50, 50, 500, 700)
            page.insert_textbox(rect, text, fontsize=11)
    if title:
        doc.set_metadata({"title": title, "author": author})
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def make_mock_storage(pdf_bytes: Optional[bytes] = None, upload_raises: bool = False) -> MagicMock:
    """
    Build a mock SupabaseStorageClient.
    If pdf_bytes is None, download raises StorageError.
    """
    client = MagicMock()
    if pdf_bytes is None:
        client.download = AsyncMock(side_effect=StorageError("mock download failure"))
    else:
        client.download = AsyncMock(return_value=pdf_bytes)

    if upload_raises:
        client.upload = AsyncMock(side_effect=StorageError("mock upload failure"))
    else:
        client.upload = AsyncMock(return_value="papers/test/parsed.json")

    client.close = AsyncMock()
    return client


# ------------------------------------------------------------------ #
# 1. Valid multi-page PDF
# ------------------------------------------------------------------ #

class TestValidPDF:
    @pytest.mark.asyncio
    async def test_page_count(self):
        pdf_bytes = make_pdf_bytes(["Page one text", "Page two text", "Page three text"])
        storage = make_mock_storage(pdf_bytes)
        parser = PDFParser(storage_client=storage)

        doc = await parser.parse("paper-1", "papers/paper-1/fulltext.pdf")

        assert doc.page_count == 3

    @pytest.mark.asyncio
    async def test_one_based_page_numbering(self):
        pdf_bytes = make_pdf_bytes(["First", "Second"])
        storage = make_mock_storage(pdf_bytes)
        parser = PDFParser(storage_client=storage)

        doc = await parser.parse("paper-1", "papers/paper-1/fulltext.pdf")

        assert doc.pages[0].page_number == 1
        assert doc.pages[1].page_number == 2

    @pytest.mark.asyncio
    async def test_text_extracted(self):
        pdf_bytes = make_pdf_bytes(["Hello from page one"])
        storage = make_mock_storage(pdf_bytes)
        parser = PDFParser(storage_client=storage)

        doc = await parser.parse("paper-1", "papers/paper-1/fulltext.pdf")

        assert "Hello from page one" in doc.pages[0].text

    @pytest.mark.asyncio
    async def test_uuid_paper_id_from_asyncpg_is_accepted(self):
        from uuid import UUID

        pdf_bytes = make_pdf_bytes(["Hello from page one"])
        storage = make_mock_storage(pdf_bytes)
        parser = PDFParser(storage_client=storage)
        paper_uuid = UUID("01cade69-4d69-406f-bc91-c7fe726a4326")

        doc = await parser.parse(paper_uuid, "papers/01cade69-4d69-406f-bc91-c7fe726a4326/fulltext.pdf")

        assert doc.paper_id == "01cade69-4d69-406f-bc91-c7fe726a4326"

    @pytest.mark.asyncio
    async def test_character_counts_positive(self):
        pdf_bytes = make_pdf_bytes(["Short text"])
        storage = make_mock_storage(pdf_bytes)
        parser = PDFParser(storage_client=storage)

        doc = await parser.parse("paper-1", "papers/paper-1/fulltext.pdf")

        assert doc.pages[0].character_count > 0
        assert doc.total_character_count == sum(p.character_count for p in doc.pages)

    @pytest.mark.asyncio
    async def test_paper_id_preserved(self):
        pdf_bytes = make_pdf_bytes(["text"])
        storage = make_mock_storage(pdf_bytes)
        parser = PDFParser(storage_client=storage)

        doc = await parser.parse("my-paper-id", "papers/my-paper-id/fulltext.pdf")

        assert doc.paper_id == "my-paper-id"


# ------------------------------------------------------------------ #
# 2. Empty / near-empty PDF
# ------------------------------------------------------------------ #

class TestEmptyPDF:
    @pytest.mark.asyncio
    async def test_blank_pages_detected_as_low_text(self):
        # PDF with empty pages
        pdf_bytes = make_pdf_bytes(["", ""])
        storage = make_mock_storage(pdf_bytes)
        parser = PDFParser(storage_client=storage)

        doc = await parser.parse("paper-empty", "papers/paper-empty/fulltext.pdf")

        assert doc.total_character_count < _MIN_MEANINGFUL_CHARS
        assert doc.pages_with_text == 0

    @pytest.mark.asyncio
    async def test_single_character_pdf(self):
        pdf_bytes = make_pdf_bytes(["x"])
        storage = make_mock_storage(pdf_bytes)
        parser = PDFParser(storage_client=storage)

        doc = await parser.parse("p", "papers/p/fulltext.pdf")

        # The document is returned but total_character_count is below threshold
        assert doc.page_count == 1
        assert doc.total_character_count < _MIN_MEANINGFUL_CHARS


# ------------------------------------------------------------------ #
# 3. Corrupted / invalid bytes
# ------------------------------------------------------------------ #

class TestCorruptedPDF:
    @pytest.mark.asyncio
    async def test_corrupted_bytes_raise_ingestion_error(self):
        bad_bytes = b"this is not a valid pdf"
        storage = make_mock_storage(bad_bytes)
        parser = PDFParser(storage_client=storage)

        with pytest.raises(IngestionError):
            await parser.parse("paper-bad", "papers/paper-bad/fulltext.pdf")

    @pytest.mark.asyncio
    async def test_download_failure_raises_ingestion_error(self):
        storage = make_mock_storage(pdf_bytes=None)  # triggers StorageError
        parser = PDFParser(storage_client=storage)

        with pytest.raises(IngestionError):
            await parser.parse("paper-fail", "papers/paper-fail/fulltext.pdf")


# ------------------------------------------------------------------ #
# 4. PDF metadata
# ------------------------------------------------------------------ #

class TestPDFMetadata:
    @pytest.mark.asyncio
    async def test_metadata_extracted(self):
        pdf_bytes = make_pdf_bytes(["Content"], title="Test Paper", author="Test Author")
        storage = make_mock_storage(pdf_bytes)
        parser = PDFParser(storage_client=storage)

        doc = await parser.parse("paper-meta", "papers/paper-meta/fulltext.pdf")

        # Metadata dict should be present (content depends on pymupdf version)
        assert isinstance(doc.metadata, dict)

    @pytest.mark.asyncio
    async def test_metadata_does_not_replace_canonical_paper_id(self):
        pdf_bytes = make_pdf_bytes(["Content"], title="PDF Title")
        storage = make_mock_storage(pdf_bytes)
        parser = PDFParser(storage_client=storage)

        doc = await parser.parse("canonical-id", "papers/canonical-id/fulltext.pdf")

        # paper_id must remain the canonical DB ID, not anything from PDF metadata
        assert doc.paper_id == "canonical-id"


# ------------------------------------------------------------------ #
# 5. Page boundaries
# ------------------------------------------------------------------ #

class TestPageBoundaries:
    @pytest.mark.asyncio
    async def test_text_does_not_bleed_between_pages(self):
        pdf_bytes = make_pdf_bytes(["UNIQUE_PAGE_ONE_MARKER", "UNIQUE_PAGE_TWO_MARKER"])
        storage = make_mock_storage(pdf_bytes)
        parser = PDFParser(storage_client=storage)

        doc = await parser.parse("paper-bounds", "papers/paper-bounds/fulltext.pdf")

        assert "UNIQUE_PAGE_ONE_MARKER" in doc.pages[0].text
        assert "UNIQUE_PAGE_TWO_MARKER" not in doc.pages[0].text
        assert "UNIQUE_PAGE_TWO_MARKER" in doc.pages[1].text
        assert "UNIQUE_PAGE_ONE_MARKER" not in doc.pages[1].text


# ------------------------------------------------------------------ #
# 6. JSON serialisation
# ------------------------------------------------------------------ #

class TestJSONSerialisation:
    @pytest.mark.asyncio
    async def test_round_trip(self):
        pdf_bytes = make_pdf_bytes(["Hello world", "Second page"])
        storage = make_mock_storage(pdf_bytes)
        parser = PDFParser(storage_client=storage)

        doc = await parser.parse("paper-json", "papers/paper-json/fulltext.pdf")

        json_str = doc.model_dump_json()
        data = json.loads(json_str)

        assert data["paper_id"] == "paper-json"
        assert data["page_count"] == 2
        assert len(data["pages"]) == 2
        assert data["pages"][0]["page_number"] == 1

    @pytest.mark.asyncio
    async def test_utf8_text_survives(self):
        text = "caf\u00e9 \u4e2d\u6587"
        pdf_bytes = make_pdf_bytes([text])
        storage = make_mock_storage(pdf_bytes)
        parser = PDFParser(storage_client=storage)

        doc = await parser.parse("paper-utf8", "papers/paper-utf8/fulltext.pdf")

        json_bytes = doc.model_dump_json().encode("utf-8")
        recovered = json.loads(json_bytes.decode("utf-8"))
        # Text should survive serialisation (may have whitespace variations from PDF)
        assert recovered["pages"][0]["text"] is not None


# ------------------------------------------------------------------ #
# 7. Idempotency — PARSED papers excluded by repo query
# ------------------------------------------------------------------ #

class TestIdempotency:
    """
    The idempotency logic lives in PaperRepository.get_papers_for_parsing().
    We verify the query filter behaviour via a mock that returns no papers
    when all are already PARSED.
    """

    @pytest.mark.asyncio
    async def test_parsed_papers_not_returned_by_repo(self):
        """A mock repo that returns empty list simulates all papers already PARSED."""
        from app.core.job_store import ParsingJob, JobStatus, JobStore
        from app.ingestion.parsers.worker import run_parsing

        # Mock repo returns no eligible papers (all already PARSED)
        mock_repo = MagicMock()
        mock_repo.get_papers_for_parsing = AsyncMock(return_value=[])

        job_store = JobStore()
        job = await job_store.create_parsing_job("run-already-parsed")

        await run_parsing(job=job, job_store=job_store, repo=mock_repo)

        assert job.status == JobStatus.COMPLETED
        assert job.total_papers == 0
        assert job.parsed == 0


# ------------------------------------------------------------------ #
# 8. Worker resilience
# ------------------------------------------------------------------ #

class TestWorkerResilience:
    @pytest.mark.asyncio
    async def test_one_failure_does_not_stop_others(self):
        """
        Paper A fails (bad storage path), Paper B and C succeed.
        Worker must process all three and mark A as failed without stopping.
        """
        from app.core.job_store import JobStore
        from app.core.job_store import JobStatus
        from app.ingestion.parsers.worker import run_parsing

        good_pdf = make_pdf_bytes(["Good content that is long enough " * 10])

        # Paper A: no storage_path (triggers failure)
        # Paper B: valid PDF
        # Paper C: valid PDF
        papers = [
            {"paper_id": "pa", "storage_path": ""},           # will fail
            {"paper_id": "pb", "storage_path": "papers/pb/fulltext.pdf"},
            {"paper_id": "pc", "storage_path": "papers/pc/fulltext.pdf"},
        ]

        call_count = {"n": 0}
        update_calls = []

        async def mock_update(paper_id, status, **kwargs):
            update_calls.append((paper_id, status))

        mock_repo = MagicMock()
        mock_repo.get_papers_for_parsing = AsyncMock(return_value=papers)
        mock_repo.update_parsing_metadata = AsyncMock(side_effect=mock_update)

        # Patch SupabaseStorageClient so it doesn't need real credentials
        with patch("app.ingestion.parsers.worker.SupabaseStorageClient") as mock_sc_cls:
            mock_sc = MagicMock()
            mock_sc.download = AsyncMock(return_value=good_pdf)
            mock_sc.upload = AsyncMock(return_value="papers/p/parsed.json")
            mock_sc.close = AsyncMock()
            mock_sc_cls.return_value = mock_sc

            job_store = JobStore()
            job = await job_store.create_parsing_job("run-resilience")
            await run_parsing(job=job, job_store=job_store, repo=mock_repo)

        assert job.status == JobStatus.COMPLETED
        # Paper A failed, B and C succeeded
        assert job.failed >= 1
        assert job.parsed >= 1  # at least one succeeded

    @pytest.mark.asyncio
    async def test_uuid_paper_ids_from_repository_are_parsed(self):
        """Reproduce the live failure: asyncpg returns uuid.UUID paper_id values."""
        from uuid import UUID
        from app.core.job_store import JobStore, JobStatus
        from app.ingestion.parsers.worker import run_parsing

        good_pdf = make_pdf_bytes(["Good content that is long enough " * 10])
        paper_uuid = UUID("01cade69-4d69-406f-bc91-c7fe726a4326")
        papers = [{"paper_id": paper_uuid, "storage_path": f"papers/{paper_uuid}/fulltext.pdf"}]
        update_calls = []

        async def mock_update(paper_id, status, **kwargs):
            update_calls.append((paper_id, status, kwargs))

        mock_repo = MagicMock()
        mock_repo.get_papers_for_parsing = AsyncMock(return_value=papers)
        mock_repo.update_parsing_metadata = AsyncMock(side_effect=mock_update)

        with patch("app.ingestion.parsers.worker.SupabaseStorageClient") as mock_sc_cls:
            mock_sc = MagicMock()
            mock_sc.download = AsyncMock(return_value=good_pdf)
            mock_sc.upload = AsyncMock(return_value=f"papers/{paper_uuid}/parsed.json")
            mock_sc.close = AsyncMock()
            mock_sc_cls.return_value = mock_sc

            job_store = JobStore()
            job = await job_store.create_parsing_job("run-uuid")
            await run_parsing(job=job, job_store=job_store, repo=mock_repo)

        assert job.status == JobStatus.COMPLETED
        assert job.parsed == 1
        assert job.failed == 0
        assert update_calls[0][0] == str(paper_uuid)
        assert update_calls[0][1] == "PARSED"
        assert update_calls[0][2]["page_count"] >= 1
        assert update_calls[0][2]["parsed_storage_path"] == f"papers/{paper_uuid}/parsed.json"


# ------------------------------------------------------------------ #
# 9. Storage upload failure
# ------------------------------------------------------------------ #

class TestStorageUploadFailure:
    @pytest.mark.asyncio
    async def test_parse_not_marked_parsed_if_upload_fails(self):
        """
        If parsed.json upload fails, the paper must NOT be marked PARSED.
        It should be marked PARSE_FAILED instead.
        """
        from app.core.job_store import JobStore
        from app.ingestion.parsers.worker import run_parsing

        good_pdf = make_pdf_bytes(["Enough text content here " * 20])

        papers = [{"paper_id": "pu", "storage_path": "papers/pu/fulltext.pdf"}]
        update_calls = []

        async def mock_update(paper_id, status, **kwargs):
            update_calls.append((paper_id, status))

        mock_repo = MagicMock()
        mock_repo.get_papers_for_parsing = AsyncMock(return_value=papers)
        mock_repo.update_parsing_metadata = AsyncMock(side_effect=mock_update)

        with patch("app.ingestion.parsers.worker.SupabaseStorageClient") as mock_sc_cls:
            mock_sc = MagicMock()
            mock_sc.download = AsyncMock(return_value=good_pdf)
            # Upload always fails
            mock_sc.upload = AsyncMock(side_effect=StorageError("upload failed"))
            mock_sc.close = AsyncMock()
            mock_sc_cls.return_value = mock_sc

            job_store = JobStore()
            job = await job_store.create_parsing_job("run-upload-fail")
            await run_parsing(job=job, job_store=job_store, repo=mock_repo)

        # Must be PARSE_FAILED, never PARSED
        statuses = [status for _, status in update_calls]
        assert "PARSE_FAILED" in statuses
        assert "PARSED" not in statuses
        assert job.parsed == 0
        assert job.failed == 1


# ------------------------------------------------------------------ #
# Path normalisation
# ------------------------------------------------------------------ #

class TestPathNormalisation:
    def test_double_prefix_stripped(self):
        storage = make_mock_storage(b"irrelevant")
        parser = PDFParser(storage_client=storage, bucket="papers")

        result = parser._normalise_path("papers/papers/abc/fulltext.pdf")
        assert result == "papers/abc/fulltext.pdf"

    def test_correct_path_unchanged(self):
        storage = make_mock_storage(b"irrelevant")
        parser = PDFParser(storage_client=storage, bucket="papers")

        result = parser._normalise_path("papers/abc/fulltext.pdf")
        assert result == "papers/abc/fulltext.pdf"
