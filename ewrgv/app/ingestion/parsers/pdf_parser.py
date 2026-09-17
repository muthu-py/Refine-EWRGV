"""
app/ingestion/parsers/pdf_parser.py
-------------------------------------
PDF text extraction for Phase 2.5 — Document Parsing.

Responsibility: Given a paper_id and its PDF storage path, download the PDF
from Supabase Storage, extract page-level text using PyMuPDF (fitz), and
return a structured ParsedDocument.

This module is deliberately narrow:
  - It does NOT access the database.
  - It does NOT upload anything.
  - It does NOT interpret or classify text.
  - It does NOT perform OCR.
  - It does NOT call external academic APIs.

All orchestration (download, parse, upload parsed.json, update DB) is
handled by the parsing worker (app/ingestion/parsers/worker.py).
"""

from __future__ import annotations

import io
from typing import Any

from app.core.exceptions import IngestionError
from app.core.logging import get_logger
from app.domain.models.parsed_document import ParsedDocument, ParsedPage
from app.storage.supabase_storage import SupabaseStorageClient

logger = get_logger(__name__)

# Minimum total character count for a document to be considered parseable.
# Documents below this threshold (e.g. scanned PDFs with no text layer,
# corrupted files, cover-page-only PDFs) are flagged as low-text.
_MIN_MEANINGFUL_CHARS = 100


class PDFParser:
    """
    Extracts page-level text from an acquired PDF stored in Supabase Storage.

    Parameters
    ----------
    storage_client:
        An initialised SupabaseStorageClient.  The caller is responsible
        for lifecycle management (i.e. calling storage_client.close()).
    bucket:
        Supabase Storage bucket name.  Defaults to "papers".
    """

    def __init__(
        self,
        storage_client: SupabaseStorageClient,
        bucket: str = "papers",
    ) -> None:
        self._storage = storage_client
        self._bucket = bucket

    async def parse(self, paper_id: str, storage_path: str) -> ParsedDocument:
        """
        Download and parse the PDF at `storage_path`.

        Parameters
        ----------
        paper_id:
            UUID of the canonical paper record.
        storage_path:
            Bucket-relative path to the PDF (e.g. "papers/{id}/fulltext.pdf").
            If the path is prefixed with the bucket name (legacy double-prefix
            from Phase 2.4 — e.g. "papers/papers/{id}/fulltext.pdf"), the
            leading "{bucket}/" prefix is stripped automatically.

        Returns
        -------
        ParsedDocument:
            Fully populated parsed document with page-level text.

        Raises
        ------
        IngestionError:
            - If the PDF cannot be downloaded from Storage.
            - If the file cannot be opened as a valid PDF.
            - If the PDF has zero pages.
        """
        # Normalise path: strip leading "{bucket}/" double-prefix if present
        clean_path = self._normalise_path(storage_path)

        logger.info(
            "PDFParser: downloading PDF",
            extra={"paper_id": paper_id, "path": clean_path},
        )

        # --- 1. Download PDF bytes ---
        try:
            pdf_bytes = await self._storage.download(self._bucket, clean_path)
        except Exception as exc:
            raise IngestionError(
                f"Failed to download PDF for paper {paper_id}: {exc}",
                detail=f"storage_path={clean_path}",
            ) from exc

        # --- 2. Open PDF with PyMuPDF ---
        doc = self._open_pdf(paper_id, pdf_bytes)

        try:
            # --- 3. Extract metadata ---
            pdf_metadata = self._extract_metadata(doc)

            # --- 4. Extract pages ---
            if doc.page_count == 0:
                raise IngestionError(
                    f"PDF has zero pages: paper_id={paper_id}",
                    detail=f"storage_path={clean_path}",
                )

            pages: list[ParsedPage] = []
            for page_index in range(doc.page_count):
                page = doc.load_page(page_index)
                text = page.get_text("text")  # plain text extraction
                pages.append(ParsedPage.from_text(page_index + 1, text))  # 1-based

            # --- 5. Build ParsedDocument ---
            parsed = ParsedDocument.build(
                paper_id=paper_id,
                source_storage_path=clean_path,
                pages=pages,
                metadata=pdf_metadata,
            )

        finally:
            doc.close()

        # --- 6. Log extraction statistics ---
        logger.info(
            "PDFParser: extraction complete",
            extra={
                "paper_id": paper_id,
                "page_count": parsed.page_count,
                "total_chars": parsed.total_character_count,
                "pages_with_text": parsed.pages_with_text,
                "low_text": parsed.total_character_count < _MIN_MEANINGFUL_CHARS,
            },
        )

        return parsed

    # ------------------------------------------------------------------ #
    # Property
    # ------------------------------------------------------------------ #

    @property
    def min_meaningful_chars(self) -> int:
        """Minimum character count considered meaningful for a parsed document."""
        return _MIN_MEANINGFUL_CHARS

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _normalise_path(self, storage_path: str) -> str:
        """
        Strip double ``{bucket}/{bucket}/`` prefix if present.

        Phase 2.4 (fulltext.py) stored paths as ``papers/papers/{id}/fulltext.pdf``
        (bucket name prepended to the bucket-relative path).  This method corrects
        that by stripping exactly one leading ``{bucket}/`` when the path starts
        with ``{bucket}/{bucket}/``.

        Examples
        --------
        "papers/papers/abc/fulltext.pdf" -> "papers/abc/fulltext.pdf"
        "papers/abc/fulltext.pdf"        -> "papers/abc/fulltext.pdf"  (unchanged)
        "abc/fulltext.pdf"               -> "abc/fulltext.pdf"         (unchanged)
        """
        double_prefix = f"{self._bucket}/{self._bucket}/"
        if storage_path.startswith(double_prefix):
            return storage_path[len(f"{self._bucket}/"):]
        return storage_path

    def _open_pdf(self, paper_id: str, pdf_bytes: bytes) -> Any:
        """Open PDF bytes with PyMuPDF. Raises IngestionError on failure."""
        try:
            import pymupdf  # noqa: PLC0415
            doc = pymupdf.open(stream=io.BytesIO(pdf_bytes), filetype="pdf")
            return doc
        except Exception as exc:
            raise IngestionError(
                f"Cannot open PDF for paper {paper_id}: {exc}",
                detail="The file may be corrupted or not a valid PDF.",
            ) from exc

    @staticmethod
    def _extract_metadata(doc: Any) -> dict[str, Any]:
        """
        Extract PDF-level metadata from a fitz Document.

        Returns an empty dict if metadata is unavailable or malformed.
        Only string-safe values are included; no binary content.
        """
        try:
            raw: dict = doc.metadata or {}
            # Keep only non-empty string values; skip binary/None entries
            return {
                k: v
                for k, v in raw.items()
                if isinstance(v, str) and v.strip()
            }
        except Exception:  # noqa: BLE001
            return {}
