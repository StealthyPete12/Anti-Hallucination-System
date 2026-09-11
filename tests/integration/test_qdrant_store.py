"""Integration tests for QdrantVectorStore against a real qdrant-client.

Uses qdrant-client's built-in ":memory:" mode, which runs the real Qdrant
search/storage engine in-process. This exercises the actual wire-format
contract (point IDs, payload shape, filters, distance metric) without
requiring a running Qdrant server or Docker - so it's still fast enough to
run by default, but is kept out of tests/unit because it goes through the
real client library rather than a fake.
"""

from __future__ import annotations

import pytest

from rag.models import Chunk
from rag.stores.qdrant import QdrantVectorStore

pytestmark = pytest.mark.integration


@pytest.fixture
def store() -> QdrantVectorStore:
    return QdrantVectorStore(url=":memory:", collection_name="test_collection")


def make_chunk(chunk_id: str, doc_id: str, text: str = "some text") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        chunk_index=int(chunk_id.split("#")[1]),
        text=text,
        source_path=f"data/corpus/{doc_id}.txt",
        heading="H",
    )


def test_collection_does_not_exist_initially(store: QdrantVectorStore) -> None:
    assert store.collection_exists() is False


def test_create_collection_makes_it_exist(store: QdrantVectorStore) -> None:
    store.create_collection(vector_size=4)
    assert store.collection_exists() is True


def test_create_collection_is_idempotent(store: QdrantVectorStore) -> None:
    store.create_collection(vector_size=4)
    store.create_collection(vector_size=4)  # should not raise
    assert store.collection_exists() is True


def test_upsert_and_search_round_trip(store: QdrantVectorStore) -> None:
    store.create_collection(vector_size=3)
    chunks = [
        make_chunk("doc#0", "doc", text="first chunk"),
        make_chunk("doc#1", "doc", text="second chunk"),
    ]
    vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    store.upsert(chunks, vectors)

    results = store.search(query_vector=[1.0, 0.0, 0.0], top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "doc#0"
    assert results[0].text == "first chunk"
    assert results[0].doc_id == "doc"
    assert results[0].heading == "H"
    assert results[0].score >= results[1].score


def test_search_respects_top_k(store: QdrantVectorStore) -> None:
    store.create_collection(vector_size=2)
    chunks = [make_chunk(f"doc#{i}", "doc") for i in range(5)]
    vectors = [[float(i), 0.0] for i in range(5)]
    store.upsert(chunks, vectors)

    results = store.search(query_vector=[0.0, 0.0], top_k=2)
    assert len(results) == 2


def test_upsert_is_idempotent_for_same_chunk_id(store: QdrantVectorStore) -> None:
    store.create_collection(vector_size=3)
    chunk = make_chunk("doc#0", "doc", text="version 1")
    store.upsert([chunk], [[1.0, 0.0, 0.0]])
    updated = make_chunk("doc#0", "doc", text="version 2")
    store.upsert([updated], [[1.0, 0.0, 0.0]])

    results = store.search(query_vector=[1.0, 0.0, 0.0], top_k=10)
    assert len(results) == 1
    assert results[0].text == "version 2"


def test_search_with_filter_restricts_results(store: QdrantVectorStore) -> None:
    store.create_collection(vector_size=2)
    chunks = [make_chunk("a#0", "a"), make_chunk("b#0", "b")]
    store.upsert(chunks, [[1.0, 0.0], [1.0, 0.0]])

    results = store.search(query_vector=[1.0, 0.0], top_k=10, filters={"doc_id": "a"})
    assert len(results) == 1
    assert results[0].doc_id == "a"


def test_delete_collection_removes_it(store: QdrantVectorStore) -> None:
    store.create_collection(vector_size=2)
    store.delete_collection()
    assert store.collection_exists() is False


def test_upsert_length_mismatch_raises() -> None:
    store = QdrantVectorStore(url=":memory:", collection_name="c")
    store.create_collection(vector_size=2)
    with pytest.raises(ValueError):
        store.upsert([make_chunk("a#0", "a")], [[1.0, 0.0], [2.0, 0.0]])


def test_upsert_empty_list_is_noop(store: QdrantVectorStore) -> None:
    store.create_collection(vector_size=2)
    store.upsert([], [])  # should not raise
