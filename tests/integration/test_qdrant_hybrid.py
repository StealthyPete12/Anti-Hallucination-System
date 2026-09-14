"""Integration tests for Phase 4 hybrid retrieval against a real qdrant-client.

Same ":memory:" real-engine approach as tests/integration/test_qdrant_store.py,
with ``enable_sparse=True`` so these exercise the actual named dense+sparse
vector wire format and RRF fusion against Qdrant's real query engine.
"""

from __future__ import annotations

import pytest

from rag.models import Chunk, SparseVector
from rag.sparse import BM25Vectorizer
from rag.stores.qdrant import QdrantVectorStore, VectorStoreError

pytestmark = pytest.mark.integration


@pytest.fixture
def store() -> QdrantVectorStore:
    return QdrantVectorStore(url=":memory:", collection_name="hybrid_test", enable_sparse=True)


def make_chunk(chunk_id: str, doc_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        chunk_index=int(chunk_id.split("#")[1]),
        text=text,
        source_path=f"data/corpus/{doc_id}.txt",
        heading="H",
    )


CORPUS = [
    make_chunk("doc#0", "doc", "PostgreSQL uses multi-version concurrency control for isolation."),
    make_chunk("doc#1", "doc", "Vacuum reclaims storage occupied by dead tuples."),
    make_chunk("doc#2", "doc", "Indexes speed up query lookups on large tables."),
]


def _index(store: QdrantVectorStore) -> BM25Vectorizer:
    store.create_collection(vector_size=3)
    # Fake, hand-distinguishable dense vectors: one-hot per chunk.
    dense_vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    store.upsert(CORPUS, dense_vectors)

    bm25 = BM25Vectorizer().fit([c.text for c in CORPUS])
    sparse_vectors = [bm25.encode_document(c.text) for c in CORPUS]
    store.upsert_sparse(CORPUS, sparse_vectors)
    return bm25


def test_dense_search_still_works_when_sparse_enabled(store: QdrantVectorStore) -> None:
    _index(store)
    results = store.search([1.0, 0.0, 0.0], top_k=1)
    assert results[0].chunk_id == "doc#0"


def test_sparse_search_finds_the_lexically_relevant_chunk(store: QdrantVectorStore) -> None:
    bm25 = _index(store)
    query = bm25.encode_query("vacuum dead tuples")
    results = store.search_sparse(query, top_k=1)
    assert results[0].chunk_id == "doc#1"


def test_search_sparse_without_enable_sparse_raises() -> None:
    plain_store = QdrantVectorStore(url=":memory:", collection_name="dense_only")
    plain_store.create_collection(vector_size=3)
    with pytest.raises(VectorStoreError):
        plain_store.search_sparse(SparseVector(indices=[0], values=[1.0]), top_k=1)


def test_hybrid_search_combines_both_signals(store: QdrantVectorStore) -> None:
    bm25 = _index(store)
    # Dense query points at doc#2; sparse query lexically matches doc#1.
    # Hybrid (RRF-fused) should surface both ahead of the untouched doc#0.
    dense_query = [0.0, 0.0, 1.0]
    sparse_query = bm25.encode_query("vacuum dead tuples")
    results = store.search_hybrid(dense_query, sparse_query, top_k=3, fanout=3)
    result_ids = [c.chunk_id for c in results]
    assert set(result_ids[:2]) == {"doc#1", "doc#2"}
    assert result_ids[2] == "doc#0"


def test_hybrid_search_scores_are_fused_rrf_scores(store: QdrantVectorStore) -> None:
    bm25 = _index(store)
    results = store.search_hybrid(
        [1.0, 0.0, 0.0], bm25.encode_query("postgresql concurrency"), top_k=3, fanout=3
    )
    # RRF scores at k=60 for a doc appearing in both branches near the top
    # are on the order of 1/61 + 1/61 ~= 0.033 - nowhere near a raw cosine
    # similarity or BM25 weight, proving the store overwrote .score with RRF.
    assert all(0 < r.score < 1 for r in results)


def test_upsert_sparse_without_enable_sparse_raises() -> None:
    plain_store = QdrantVectorStore(url=":memory:", collection_name="dense_only_2")
    plain_store.create_collection(vector_size=2)
    chunk = make_chunk("a#0", "a", "text")
    plain_store.upsert([chunk], [[1.0, 0.0]])
    with pytest.raises(VectorStoreError):
        plain_store.upsert_sparse([chunk], [SparseVector(indices=[0], values=[1.0])])
