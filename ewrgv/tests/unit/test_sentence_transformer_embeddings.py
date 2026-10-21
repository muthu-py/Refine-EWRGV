"""
tests/unit/test_sentence_transformer_embeddings.py
----------------------------------------------------
Local sentence-transformers provider, with a fake SentenceTransformer so no
model is downloaded.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from app.core.config import settings
from app.core.exceptions import ProviderError
from app.providers.embeddings import factory
from app.providers.embeddings import sentence_transformer_embeddings as ste


class FakeSentenceTransformer:
    loads: list[str] = []

    def __init__(self, model_name: str) -> None:
        FakeSentenceTransformer.loads.append(model_name)
        self.calls: list[dict] = []

    def encode(self, texts, **kwargs):
        self.calls.append({"texts": list(texts), **kwargs})
        dim = 384 if "small" in FakeSentenceTransformer.loads[-1] else 8
        return np.array([[float(len(t))] + [0.0] * (dim - 1) for t in texts])


@pytest.fixture(autouse=True)
def fake_st(monkeypatch):
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setattr(ste, "_MODELS", {})
    FakeSentenceTransformer.loads = []


def test_embed_batch_normalises_and_keeps_order() -> None:
    p = ste.SentenceTransformerEmbeddingProvider(model="bge-small", dimension=384, batch_size=7)
    vectors = p.embed_batch(["a", "abc"])
    assert [v[0] for v in vectors] == [1.0, 3.0]
    assert all(len(v) == 384 for v in vectors)
    call = ste._MODELS["bge-small"].calls[0]
    assert call["normalize_embeddings"] is True
    assert call["batch_size"] == 7


def test_model_loads_once_and_is_shared() -> None:
    a = ste.SentenceTransformerEmbeddingProvider(model="bge-small", dimension=384)
    b = ste.SentenceTransformerEmbeddingProvider(model="bge-small", dimension=384)
    assert FakeSentenceTransformer.loads == []  # lazy
    a.embed("x")
    b.embed("y")
    assert FakeSentenceTransformer.loads == ["bge-small"]


@pytest.mark.asyncio
async def test_async_methods() -> None:
    p = ste.SentenceTransformerEmbeddingProvider(model="bge-small", dimension=384)
    assert len(await p.embed_text("hello")) == 384
    assert len(await p.embed_documents(["a", "b", "c"])) == 3
    assert await p.embed_documents([]) == []


def test_dimension_mismatch_is_reported() -> None:
    p = ste.SentenceTransformerEmbeddingProvider(model="other", dimension=384)
    with pytest.raises(ProviderError, match="8-dim vectors but EMBEDDING_DIMENSION=384"):
        p.embed("x")


def test_metadata() -> None:
    p = ste.SentenceTransformerEmbeddingProvider(model="bge-small", dimension=384)
    assert p.get_model_name() == "bge-small"
    assert p.get_dimension() == 384


@pytest.mark.parametrize("name", ["sentence-transformers", "local"])
def test_factory_builds_local_provider_without_key(monkeypatch, name) -> None:
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", name)
    monkeypatch.setattr(settings, "EMBEDDING_API_KEY", "")
    assert factory.embedding_provider_configured() is True
    provider = factory.create_embedding_provider(model="bge-small", dimension=384)
    assert isinstance(provider, ste.SentenceTransformerEmbeddingProvider)


def test_api_provider_still_needs_key(monkeypatch) -> None:
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(settings, "EMBEDDING_API_KEY", "")
    assert factory.embedding_provider_configured() is False
