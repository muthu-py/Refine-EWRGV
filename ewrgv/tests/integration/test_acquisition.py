"""
tests/integration/test_acquisition.py
"""
import pytest
from unittest.mock import AsyncMock, patch

from app.ingestion.acquisition.discovery import OADiscoveryService
from app.ingestion.acquisition.fulltext import FullTextAcquisitionService
from app.core.config import settings

@pytest.fixture
def mock_httpx_client():
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client_cls.return_value = mock_client
        
        # Add aclose method
        mock_client.aclose = AsyncMock()
        
        yield mock_client

@pytest.mark.asyncio
async def test_oa_discovery_already_has_url():
    service = OADiscoveryService()
    paper = {
        "full_text_url": "https://example.com/pdf",
        "source_type": "openalex",
        "doi": "10.1234/test"
    }
    
    url, source = await service.discover(paper)
    assert url == "https://example.com/pdf"
    assert source == "openalex"

@pytest.mark.asyncio
async def test_oa_discovery_unpaywall_fallback(mock_httpx_client, monkeypatch):
    monkeypatch.setattr(settings, "UNPAYWALL_EMAIL", "test@test.com")
    service = OADiscoveryService()
    
    # Mock unpaywall success response
    from unittest.mock import MagicMock
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "is_oa": True,
        "best_oa_location": {
            "url_for_pdf": "https://unpaywall.org/pdf",
            "url": "https://unpaywall.org/other"
        }
    }
    mock_httpx_client.get.return_value = mock_resp
    
    paper = {
        "full_text_url": None,
        "source_type": None,
        "doi": "10.1234/unpaywall"
    }
    
    url, source = await service.discover(paper)
    assert url == "https://unpaywall.org/pdf"
    assert source == "unpaywall"
    
    mock_httpx_client.get.assert_called_once()
    args, kwargs = mock_httpx_client.get.call_args
    assert "10.1234/unpaywall" in args[0]
    assert kwargs["params"]["email"] == settings.UNPAYWALL_EMAIL

@pytest.mark.asyncio
async def test_oa_discovery_no_oa(mock_httpx_client):
    service = OADiscoveryService()
    
    from unittest.mock import MagicMock
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "is_oa": False
    }
    mock_httpx_client.get.return_value = mock_resp
    
    paper = {
        "full_text_url": None,
        "source_type": None,
        "doi": "10.1234/nooa"
    }
    
    url, source = await service.discover(paper)
    assert url is None
    assert source is None

import contextlib

@pytest.mark.asyncio
async def test_fulltext_acquisition_success(mock_httpx_client, monkeypatch):
    monkeypatch.setattr(settings, "SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setattr(settings, "SUPABASE_SERVICE_ROLE_KEY", "test-key")
    
    service = FullTextAcquisitionService()
    
    # Mock download stream response
    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.headers = {
        "Content-Type": "application/pdf",
        "Content-Length": "1024"
    }
    mock_resp.aread.return_value = b"fake-pdf-content"
    
    @contextlib.asynccontextmanager
    async def mock_stream(*args, **kwargs):
        yield mock_resp
        
    mock_httpx_client.stream = mock_stream
    
    # Mock upload response
    mock_upload = AsyncMock()
    mock_upload.status_code = 200
    mock_httpx_client.post.return_value = mock_upload
    
    stats = await service.acquire("p123", "https://pdf.url")
    
    assert stats["status"] == "DOWNLOADED"
    assert stats["storage_path"] == f"{settings.SUPABASE_STORAGE_BUCKET}/papers/p123/fulltext.pdf"
    assert stats["content_type"] == "application/pdf"
    assert stats["file_size"] == 16
    
    # Since stream is mocked with @asynccontextmanager, we can't assert_called_once_with easily on it directly, but the upload post will be called.
    mock_httpx_client.post.assert_called_once()
    
    upload_url, = mock_httpx_client.post.call_args[0]
    upload_kwargs = mock_httpx_client.post.call_args[1]
    
    assert upload_url == f"https://test.supabase.co/storage/v1/object/{settings.SUPABASE_STORAGE_BUCKET}/papers/p123/fulltext.pdf"
    assert upload_kwargs["headers"]["Authorization"] == "Bearer test-key"
    assert upload_kwargs["headers"]["x-upsert"] == "true"
    assert upload_kwargs["content"] == b"fake-pdf-content"

@pytest.mark.asyncio
async def test_fulltext_acquisition_invalid_content(mock_httpx_client, monkeypatch):
    monkeypatch.setattr(settings, "SUPABASE_URL", "https://test.supabase.co")
    monkeypatch.setattr(settings, "SUPABASE_SERVICE_ROLE_KEY", "test-key")
    
    service = FullTextAcquisitionService()
    
    mock_resp = AsyncMock()
    mock_resp.status_code = 200
    mock_resp.headers = {
        "Content-Type": "text/html",  # INVALID
    }
    
    @contextlib.asynccontextmanager
    async def mock_stream(*args, **kwargs):
        yield mock_resp
        
    mock_httpx_client.stream = mock_stream
    
    stats = await service.acquire("p123", "https://pdf.url")
    
    assert stats["status"] == "INVALID_DOCUMENT"
    assert stats["storage_path"] is None
    
    # Upload should NOT have been called
    mock_httpx_client.post.assert_not_called()
