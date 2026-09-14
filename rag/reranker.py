"""Cross-encoder reranking: re-scores an initial retrieval candidate set
against the query and returns the top N, discarding the vector store's
own rank/score entirely for the ranking decision (Phase 4).

Pipeline shape this plugs into:

    Query -> Initial Retrieval (top `retrieval_fanout`) -> Cross-Encoder ->
    top `top_k` -> Generator

``model_name`` is config-driven (``RAG_RERANKER_MODEL``): the required
baseline is ``cross-encoder/ms-marco-MiniLM-L-6-v2``; ``BAAI/bge-reranker-base``
is supported as a drop-in alternative through the same config value, so
comparing them (section 8 of the Phase 4 spec) is a one-line settings change.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from rag.models import RetrievedChunk

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

logger = logging.getLogger("rag.reranker")

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class RerankerError(RuntimeError):
    """Raised when the cross-encoder model fails to load or score."""


class CrossEncoderReranker:
    """Reranks retrieved chunks with a ``sentence-transformers`` cross-encoder."""

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL, device: str = "cpu") -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> CrossEncoder:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RerankerError(
                "sentence-transformers is required for CrossEncoderReranker"
            ) from exc
        logger.info("Loading reranker model %s", self.model_name)
        try:
            return CrossEncoder(self.model_name, device=self.device)
        except Exception as exc:  # noqa: BLE001
            raise RerankerError(
                f"Failed to load reranker model {self.model_name!r}: {exc}"
            ) from exc

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        """Score every (query, chunk.text) pair and return the top ``top_k`` by that score.

        Returns ``chunks[:top_k]`` unchanged (no model call) if ``chunks`` is
        already empty - nothing to rerank.
        """
        if not chunks:
            return []
        pairs = [(query, chunk.text) for chunk in chunks]
        try:
            scores = self.model.predict(pairs)
        except Exception as exc:  # noqa: BLE001
            raise RerankerError(
                f"Reranker scoring failed (model={self.model_name}): {exc}"
            ) from exc

        ranked = sorted(zip(chunks, scores, strict=True), key=lambda pair: -float(pair[1]))
        top = [replace(chunk, score=float(score)) for chunk, score in ranked[:top_k]]
        logger.debug(
            "Reranked %d candidate(s) with %s -> top %d", len(chunks), self.model_name, len(top)
        )
        return top
