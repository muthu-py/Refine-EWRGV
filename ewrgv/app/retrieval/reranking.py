"""Lazy, reusable sentence-transformers cross-encoder reranking."""

from __future__ import annotations

import threading
from functools import lru_cache
from time import perf_counter
from typing import Any

import numpy as np

from app.core.config import settings
from app.domain.models.paper_chunk import PaperChunk
from app.domain.models.retrieval_result import RetrievalResult

class RerankerError(RuntimeError):
    """Raised when a cross-encoder cannot load or score a candidate batch."""


class CrossEncoderReranker:
    """Score query/passage pairs with one lazily-loaded CrossEncoder instance.

    Importing this module does not import sentence-transformers or load model
    weights. The model is initialized on the first non-empty scoring request.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        device: str = "cpu",
        batch_size: int = 16,
        model: Any | None = None,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = max(1, batch_size)
        self._model = model
        self._load_error: Exception | None = None
        self._lock = threading.RLock()

    def _get_model(self) -> Any:
        with self._lock:
            if self._model is not None:
                return self._model
            if self._load_error is not None:
                raise RerankerError("Cross-encoder model could not be loaded") from self._load_error
            try:
                from sentence_transformers import CrossEncoder  # optional dependency

                self._model = CrossEncoder(self._model_name, device=self._device)
            except Exception as exc:  # import, cache, download and device errors
                self._load_error = exc
                raise RerankerError("Cross-encoder model could not be loaded") from exc
            return self._model

    def score_pairs(self, query: str, passages: list[str]) -> list[float]:
        """Return raw model logits for query/passage pairs, in input order."""
        if not passages:
            return []
        pairs = [(query, passage) for passage in passages]
        with self._lock:
            model = self._get_model()
            try:
                scores = model.predict(
                    pairs,
                    batch_size=self._batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            except Exception as exc:
                raise RerankerError("Cross-encoder inference failed") from exc
        values = np.asarray(scores, dtype=float).reshape(-1)
        if len(values) != len(passages) or not np.isfinite(values).all():
            raise RerankerError("Cross-encoder returned invalid scores")
        return [float(score) for score in values]

    def rerank_results(
        self,
        query: str,
        results: list[RetrievalResult],
        top_n: int = 10,
    ) -> tuple[list[RetrievalResult], float]:
        """Rerank the leading candidates, preserving retrieval scores verbatim."""
        if not results or top_n <= 0:
            return results, 0.0
        count = min(top_n, len(results))
        started = perf_counter()
        scores = self.score_pairs(query, [r.text for r in results[:count]])
        elapsed_ms = (perf_counter() - started) * 1000
        scored = [
            result.model_copy(update={
                "metadata": {
                    **(result.metadata or {}),
                    "retrieval_rank": result.rank,
                    "reranker_score": score,
                }
            })
            for result, score in zip(results[:count], scores)
        ]
        scored.sort(key=lambda result: result.metadata["reranker_score"], reverse=True)
        # Candidates outside the configured scoring window retain retrieval order.
        return scored + results[count:], elapsed_ms

    def rerank_results_by_query(
        self,
        results: list[RetrievalResult],
        queries: dict[str, str],
        top_n: int = 10,
    ) -> tuple[list[RetrievalResult], float]:
        """Score fused passages against each passage's strongest retrieval query."""
        if not results or top_n <= 0:
            return results, 0.0
        count = min(top_n, len(results))
        started = perf_counter()
        pairs = [(queries[result.chunk_id], result.text) for result in results[:count]]
        with self._lock:
            model = self._get_model()
            try:
                raw = model.predict(
                    pairs,
                    batch_size=self._batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                )
            except Exception as exc:
                raise RerankerError("Cross-encoder inference failed") from exc
        scores = np.asarray(raw, dtype=float).reshape(-1)
        if len(scores) != count or not np.isfinite(scores).all():
            raise RerankerError("Cross-encoder returned invalid scores")
        elapsed_ms = (perf_counter() - started) * 1000
        scored = [
            result.model_copy(update={
                "metadata": {
                    **(result.metadata or {}),
                    "retrieval_rank": result.rank,
                    "reranker_score": float(score),
                }
            })
            for result, score in zip(results[:count], scores)
        ]
        scored.sort(key=lambda result: result.metadata["reranker_score"], reverse=True)
        return scored + results[count:], elapsed_ms

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(
        self,
        query: str,
        chunks: list[PaperChunk],
        top_n: int = 10,
    ) -> list[PaperChunk]:
        """Protocol-compatible PaperChunk adapter backed by cross-encoder scores."""
        count = min(max(0, top_n), len(chunks))
        if count == 0:
            return []
        scores = self.score_pairs(query, [chunk.text for chunk in chunks[:count]])
        ranked = sorted(zip(chunks[:count], scores), key=lambda item: item[1], reverse=True)
        return [chunk for chunk, _ in ranked]


@lru_cache(maxsize=4)
def get_cross_encoder_reranker(
    model_name: str | None = None,
    device: str | None = None,
    batch_size: int | None = None,
) -> CrossEncoderReranker:
    """Return a reusable, still-lazy reranker for the active configuration."""
    return CrossEncoderReranker(
        model_name=model_name or settings.RERANK_MODEL,
        device=device or settings.RERANK_DEVICE,
        batch_size=batch_size or settings.RERANK_BATCH_SIZE,
    )
