"""Unit tests for CrossEncoderReranker (rag/reranker.py).

The real ``sentence_transformers.CrossEncoder`` model is never loaded here -
``CrossEncoderReranker._model`` is injected directly with a small fake that
satisfies the one method actually used (``predict``), matching the fakes
pattern used throughout ``tests/unit``.
"""

from __future__ import annotations

import pytest

from rag.models import RetrievedChunk
from rag.reranker import CrossEncoderReranker, RerankerError


class FakeCrossEncoder:
    """Scores each (query, text) pair by how many query words appear in text."""

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = []
        for query, text in pairs:
            query_words = set(query.lower().split())
            text_words = set(text.lower().split())
            scores.append(float(len(query_words & text_words)))
        return scores


class RaisingCrossEncoder:
    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        raise RuntimeError("model exploded")


def make_chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("#")[0],
        text=text,
        score=0.1,
        source_path="x.txt",
        heading="H",
        chunk_index=0,
    )


def test_rerank_reorders_by_relevance() -> None:
    reranker = CrossEncoderReranker()
    reranker._model = FakeCrossEncoder()
    chunks = [
        make_chunk("a#0", "completely unrelated text"),
        make_chunk("b#0", "postgres mvcc transaction isolation"),
    ]
    result = reranker.rerank("postgres mvcc isolation", chunks, top_k=2)
    assert [c.chunk_id for c in result] == ["b#0", "a#0"]


def test_rerank_truncates_to_top_k() -> None:
    reranker = CrossEncoderReranker()
    reranker._model = FakeCrossEncoder()
    chunks = [make_chunk(f"c#{i}", f"word{i}") for i in range(10)]
    result = reranker.rerank("query", chunks, top_k=3)
    assert len(result) == 3


def test_rerank_empty_input_returns_empty_without_model_call() -> None:
    reranker = CrossEncoderReranker()
    reranker._model = RaisingCrossEncoder()  # would raise if ever called
    assert reranker.rerank("query", [], top_k=5) == []


def test_rerank_updates_score_field() -> None:
    reranker = CrossEncoderReranker()
    reranker._model = FakeCrossEncoder()
    chunks = [make_chunk("a#0", "postgres mvcc")]
    result = reranker.rerank("postgres mvcc", chunks, top_k=1)
    assert result[0].score == 2.0  # both query words match


def test_rerank_propagates_model_failure_as_reranker_error() -> None:
    reranker = CrossEncoderReranker()
    reranker._model = RaisingCrossEncoder()
    with pytest.raises(RerankerError):
        reranker.rerank("query", [make_chunk("a#0", "text")], top_k=1)


def test_model_name_is_configurable() -> None:
    reranker = CrossEncoderReranker(model_name="BAAI/bge-reranker-base")
    assert reranker.model_name == "BAAI/bge-reranker-base"
