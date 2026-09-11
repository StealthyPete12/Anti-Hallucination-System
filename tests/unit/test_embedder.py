"""Unit tests for LocalBGEEmbedder (rag/embedders/local.py).

These tests never download or load real model weights: a fake stand-in is
injected directly into the private ``_model`` slot so ``.model`` skips lazy
loading. This keeps the suite fast and network-independent while still
exercising the real query-prefix and batching logic.
"""

from __future__ import annotations

from typing import Any

from rag.embedders.local import BGE_QUERY_PREFIX, LocalBGEEmbedder


class _FakeVector:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def tolist(self) -> list[float]:
        return self._values


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return text.split()


class _FakeSentenceTransformer:
    """Records every text passed to .encode() so tests can assert on it."""

    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension
        self.calls: list[Any] = []
        self.tokenizer = _FakeTokenizer()

    def encode(self, texts: Any, **kwargs: Any) -> Any:
        self.calls.append(texts)
        if isinstance(texts, str):
            return _FakeVector([1.0] * self.dimension)
        return [_FakeVector([1.0] * self.dimension) for _ in texts]

    def get_embedding_dimension(self) -> int:
        return self.dimension


def make_embedder() -> tuple[LocalBGEEmbedder, _FakeSentenceTransformer]:
    embedder = LocalBGEEmbedder()
    fake = _FakeSentenceTransformer()
    embedder._model = fake  # type: ignore[assignment]
    return embedder, fake


def test_embed_documents_does_not_apply_query_prefix() -> None:
    embedder, fake = make_embedder()
    embedder.embed_documents(["some passage text", "another passage"])
    assert fake.calls[0] == ["some passage text", "another passage"]
    assert not any(BGE_QUERY_PREFIX in t for t in fake.calls[0])


def test_embed_query_applies_query_prefix() -> None:
    embedder, fake = make_embedder()
    embedder.embed_query("what is mvcc?")
    assert fake.calls[0] == f"{BGE_QUERY_PREFIX}what is mvcc?"


def test_embed_documents_empty_list_short_circuits() -> None:
    embedder, fake = make_embedder()
    assert embedder.embed_documents([]) == []
    assert fake.calls == []


def test_embed_documents_returns_plain_lists() -> None:
    embedder, _fake = make_embedder()
    vectors = embedder.embed_documents(["a", "b"])
    assert vectors == [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    assert all(isinstance(v, list) for v in vectors)


def test_dimension_reads_from_model() -> None:
    embedder, _fake = make_embedder()
    assert embedder.dimension == 4


def test_count_tokens_uses_model_tokenizer() -> None:
    embedder, _fake = make_embedder()
    assert embedder.count_tokens("one two three") == 3


def test_batch_size_is_passed_through_to_encode() -> None:
    calls: list[dict[str, Any]] = []

    class RecordingFake(_FakeSentenceTransformer):
        def encode(self, texts: Any, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return super().encode(texts, **kwargs)

    embedder = LocalBGEEmbedder(batch_size=7)
    embedder._model = RecordingFake()  # type: ignore[assignment]
    embedder.embed_documents(["a", "b"])
    assert calls[0]["batch_size"] == 7
