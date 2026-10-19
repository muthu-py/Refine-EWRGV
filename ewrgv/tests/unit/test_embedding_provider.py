import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from app.core.exceptions import ProviderError
from app.providers.embeddings.openai_embeddings import OpenAIEmbeddingProvider

# ---------------------------------------------------------------------------
# Test Data & Fixtures
# ---------------------------------------------------------------------------

DUMMY_API_KEY = "sk-dummy-key"
DUMMY_MODEL = "text-embedding-3-small"
DUMMY_DIMENSION = 1536


@pytest.fixture
def provider():
    return OpenAIEmbeddingProvider(
        api_key=DUMMY_API_KEY,
        model=DUMMY_MODEL,
        dimension=DUMMY_DIMENSION,
        max_retries=1,
        retry_delay=0.01,
    )


def make_mock_response(status_code: int, data: dict = None, text: str = ""):
    response = MagicMock(spec=httpx.Response)
    response.status_code = status_code
    if data is not None:
        response.json.return_value = data
    response.text = text
    response.headers = {}
    return response


def create_valid_response(texts: list[str], dim: int = DUMMY_DIMENSION):
    data = []
    for i, _ in enumerate(texts):
        # Create a dummy vector of the correct dimension
        vec = [0.1] * dim
        vec[0] = float(i) # Make them slightly different
        data.append({"embedding": vec, "index": i})
    return {"data": data, "model": DUMMY_MODEL}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_embed_text_success(provider):
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = make_mock_response(200, create_valid_response(["test"]))
        
        result = await provider.embed_text("test text")
        
        assert len(result) == DUMMY_DIMENSION
        assert result[0] == 0.0
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert kwargs["json"]["input"] == ["test text"]
        assert kwargs["json"]["model"] == DUMMY_MODEL
        assert kwargs["headers"]["Authorization"] == f"Bearer {DUMMY_API_KEY}"


@pytest.mark.asyncio
async def test_embed_documents_success(provider):
    texts = ["doc 1", "doc 2", "doc 3"]
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = make_mock_response(200, create_valid_response(texts))
        
        results = await provider.embed_documents(texts)
        
        assert len(results) == 3
        for i, res in enumerate(results):
            assert len(res) == DUMMY_DIMENSION
            assert res[0] == float(i)


@pytest.mark.asyncio
async def test_empty_input(provider):
    results = await provider.embed_documents([])
    assert results == []


@pytest.mark.asyncio
async def test_api_key_missing(monkeypatch):
    monkeypatch.setattr("app.providers.embeddings.openai_embeddings.settings.EMBEDDING_API_KEY", "")
    p = OpenAIEmbeddingProvider(api_key="")
    with pytest.raises(ProviderError, match="Embedding API key not configured"):
        await p.embed_text("test")


@pytest.mark.asyncio
async def test_retry_on_429(provider):
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        # First call fails with 429, second succeeds
        fail_resp = make_mock_response(429, text="Rate limit exceeded")
        success_resp = make_mock_response(200, create_valid_response(["test"]))
        mock_post.side_effect = [fail_resp, success_resp]
        
        result = await provider.embed_text("test text")
        
        assert len(result) == DUMMY_DIMENSION
        assert mock_post.call_count == 2


@pytest.mark.asyncio
async def test_exhaust_retries(provider):
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = make_mock_response(500, text="Internal Server Error")
        
        with pytest.raises(ProviderError, match="Embedding API error: HTTP 500"):
            await provider.embed_text("test text")
            
        assert mock_post.call_count == provider._max_retries + 1


@pytest.mark.asyncio
async def test_non_retryable_error(provider):
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        # 400 Bad Request should not be retried
        mock_post.return_value = make_mock_response(400, text="Bad Request")
        
        with pytest.raises(ProviderError, match=r"rejected the request (HTTP 400): Bad Request"):
            await provider.embed_text("test text")
            
        assert mock_post.call_count == 1


@pytest.mark.asyncio
async def test_network_error(provider):
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.RequestError("Network unreachable")
        
        with pytest.raises(ProviderError, match="Embedding API network error"):
            await provider.embed_text("test text")
            
        assert mock_post.call_count == provider._max_retries + 1


@pytest.mark.asyncio
async def test_dimension_mismatch_correction(provider):
    # If the provider returns a different dimension than expected, it should update its dimension tracking
    texts = ["test"]
    different_dim = 768
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = make_mock_response(200, create_valid_response(texts, dim=different_dim))
        
        assert provider.get_dimension() == DUMMY_DIMENSION
        
        result = await provider.embed_text("test")
        
        assert len(result) == different_dim
        assert provider.get_dimension() == different_dim


@pytest.mark.asyncio
async def test_invalid_response_format(provider):
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = make_mock_response(200, {"wrong": "format"})
        
        with pytest.raises(ProviderError, match="Invalid embedding response: missing 'data' key"):
            await provider.embed_text("test")


def test_sync_wrappers(provider):
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = make_mock_response(200, create_valid_response(["test"]))
        
        # Test embed()
        result = provider.embed("test text")
        assert len(result) == DUMMY_DIMENSION
        
        # Test embed_batch()
        mock_post.return_value = make_mock_response(200, create_valid_response(["doc1", "doc2"]))
        results = provider.embed_batch(["doc1", "doc2"])
        assert len(results) == 2
