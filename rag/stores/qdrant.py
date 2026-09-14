"""Qdrant-backed vector store for chunk embeddings.

Point IDs are deterministic (uuid5 of the chunk_id), so re-indexing
unchanged chunks overwrites the same points instead of duplicating them.
Full chunk metadata is stored in the payload alongside the vector so it can
be returned directly from search and filtered on (e.g. restricting
retrieval to a doc_id or heading).

Phase 4 hybrid retrieval: when constructed with ``enable_sparse=True``, the
collection additionally carries a named sparse vector ("sparse", alongside
the dense "dense" named vector) so BM25 sparse search and dense+sparse RRF
fusion (:func:`rag.fusion.reciprocal_rank_fusion`) both work against the
same collection. ``enable_sparse=False`` (the default) keeps the exact
single-unnamed-vector schema Phase 1-3 always used, so existing dense-only
callers and tests are completely unaffected.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from rag.fusion import reciprocal_rank_fusion
from rag.models import Chunk, RetrievedChunk, SparseVector

logger = logging.getLogger("rag.stores.qdrant")

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class VectorStoreError(RuntimeError):
    """Raised when a Qdrant operation fails."""


def _stable_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _build_filter(filters: dict[str, Any] | None) -> qmodels.Filter | None:
    if not filters:
        return None
    conditions: list[qmodels.Condition] = [
        qmodels.FieldCondition(key=key, match=qmodels.MatchValue(value=value))
        for key, value in filters.items()
    ]
    return qmodels.Filter(must=conditions)


class QdrantVectorStore:
    """Vector store backed by a Qdrant collection.

    Dense-only by default (``enable_sparse=False``), matching the exact
    schema Phase 1-3 used. Pass ``enable_sparse=True`` to additionally
    provision a named sparse vector for BM25/hybrid retrieval (Phase 4) -
    see the module docstring.
    """

    def __init__(
        self,
        url: str,
        collection_name: str,
        timeout: float = 30.0,
        enable_sparse: bool = False,
    ) -> None:
        self.url = url
        self.collection_name = collection_name
        self.enable_sparse = enable_sparse
        try:
            self._client = QdrantClient(location=url, timeout=int(timeout))
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to connect to Qdrant at {url!r}: {exc}") from exc

    def collection_exists(self, name: str | None = None) -> bool:
        name = name or self.collection_name
        try:
            return self._client.collection_exists(name)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to check collection {name!r}: {exc}") from exc

    def create_collection(self, name: str | None = None, vector_size: int = 0) -> None:
        name = name or self.collection_name
        if vector_size <= 0:
            raise ValueError(f"vector_size must be positive, got {vector_size}")
        if self.collection_exists(name):
            logger.info("Collection %s already exists; skipping creation", name)
            return
        logger.info(
            "Creating collection %s (dim=%d, sparse=%s)", name, vector_size, self.enable_sparse
        )
        try:
            if self.enable_sparse:
                self._client.create_collection(
                    collection_name=name,
                    vectors_config={
                        DENSE_VECTOR_NAME: qmodels.VectorParams(
                            size=vector_size, distance=qmodels.Distance.COSINE
                        )
                    },
                    sparse_vectors_config={SPARSE_VECTOR_NAME: qmodels.SparseVectorParams()},
                )
            else:
                self._client.create_collection(
                    collection_name=name,
                    vectors_config=qmodels.VectorParams(
                        size=vector_size, distance=qmodels.Distance.COSINE
                    ),
                )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to create collection {name!r}: {exc}") from exc

    def delete_collection(self, name: str | None = None) -> None:
        name = name or self.collection_name
        try:
            self._client.delete_collection(name)
            logger.info("Deleted collection %s", name)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to delete collection {name!r}: {exc}") from exc

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks ({len(chunks)}) and vectors ({len(vectors)}) length mismatch")
        if not chunks:
            return
        points = [
            qmodels.PointStruct(
                id=_stable_point_id(chunk.chunk_id),
                vector={DENSE_VECTOR_NAME: vector} if self.enable_sparse else vector,
                payload={"text": chunk.text, **chunk.to_metadata()},
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        try:
            self._client.upsert(collection_name=self.collection_name, points=points)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to upsert {len(points)} point(s): {exc}") from exc
        logger.info("Upserted %d point(s) into %s", len(points), self.collection_name)

    def upsert_sparse(self, chunks: list[Chunk], sparse_vectors: list[SparseVector]) -> None:
        """Attach a sparse vector to each chunk's already-upserted point (Phase 4).

        Requires ``enable_sparse=True`` and that ``upsert`` already created a
        point for each ``chunk_id`` - this only adds the named "sparse"
        vector to the existing point, it never creates a new one.
        """
        if not self.enable_sparse:
            raise VectorStoreError("upsert_sparse requires enable_sparse=True")
        if len(chunks) != len(sparse_vectors):
            raise ValueError(
                f"chunks ({len(chunks)}) and sparse_vectors ({len(sparse_vectors)}) length mismatch"
            )
        if not chunks:
            return
        points = [
            qmodels.PointVectors(
                id=_stable_point_id(chunk.chunk_id),
                vector={
                    SPARSE_VECTOR_NAME: qmodels.SparseVector(
                        indices=sparse.indices, values=sparse.values
                    )
                },
            )
            for chunk, sparse in zip(chunks, sparse_vectors, strict=True)
        ]
        try:
            self._client.update_vectors(collection_name=self.collection_name, points=points)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(
                f"Failed to upsert {len(points)} sparse vector(s): {exc}"
            ) from exc
        logger.info("Upserted %d sparse vector(s) into %s", len(points), self.collection_name)

    def _point_to_retrieved_chunk(self, point: Any) -> RetrievedChunk | None:
        if point.payload is None:
            return None
        return RetrievedChunk(
            chunk_id=point.payload["chunk_id"],
            doc_id=point.payload["doc_id"],
            text=point.payload["text"],
            score=point.score,
            source_path=point.payload.get("source_path", ""),
            heading=point.payload.get("heading", ""),
            chunk_index=point.payload.get("chunk_index", -1),
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Dense-vector search. Uses the named "dense" vector when ``enable_sparse``."""
        qfilter = _build_filter(filters)
        try:
            response = self._client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                using=DENSE_VECTOR_NAME if self.enable_sparse else None,
                limit=top_k,
                query_filter=qfilter,
                with_payload=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Search failed: {exc}") from exc

        retrieved = [
            chunk for point in response.points if (chunk := self._point_to_retrieved_chunk(point))
        ]
        logger.debug("Dense search returned %d result(s)", len(retrieved))
        return retrieved

    def search_sparse(
        self,
        sparse_query: SparseVector,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """BM25 sparse-vector search against the named "sparse" vector (Phase 4)."""
        if not self.enable_sparse:
            raise VectorStoreError("search_sparse requires enable_sparse=True")
        qfilter = _build_filter(filters)
        try:
            response = self._client.query_points(
                collection_name=self.collection_name,
                query=qmodels.SparseVector(
                    indices=sparse_query.indices, values=sparse_query.values
                ),
                using=SPARSE_VECTOR_NAME,
                limit=top_k,
                query_filter=qfilter,
                with_payload=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Sparse search failed: {exc}") from exc

        retrieved = [
            chunk for point in response.points if (chunk := self._point_to_retrieved_chunk(point))
        ]
        logger.debug("Sparse search returned %d result(s)", len(retrieved))
        return retrieved

    def search_hybrid(
        self,
        query_vector: list[float],
        sparse_query: SparseVector,
        top_k: int,
        *,
        fanout: int,
        rrf_k: int = 60,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """Dense + sparse search, fused client-side with Reciprocal Rank Fusion (Phase 4).

        Each branch retrieves ``fanout`` candidates independently; the two
        rank-ordered chunk_id lists are then merged with
        :func:`rag.fusion.reciprocal_rank_fusion`, and the fused top ``top_k``
        are returned (their ``.score`` field is overwritten with the fused
        RRF score, since dense cosine similarity and BM25 weight are not on
        comparable scales).
        """
        dense_results = self.search(query_vector, top_k=fanout, filters=filters)
        sparse_results = self.search_sparse(sparse_query, top_k=fanout, filters=filters)

        by_chunk_id: dict[str, RetrievedChunk] = {c.chunk_id: c for c in dense_results}
        for chunk in sparse_results:
            by_chunk_id.setdefault(chunk.chunk_id, chunk)

        fused = reciprocal_rank_fusion(
            [[c.chunk_id for c in dense_results], [c.chunk_id for c in sparse_results]],
            k=rrf_k,
        )

        results: list[RetrievedChunk] = []
        for chunk_id, fused_score in fused[:top_k]:
            base = by_chunk_id[chunk_id]
            results.append(
                RetrievedChunk(
                    chunk_id=base.chunk_id,
                    doc_id=base.doc_id,
                    text=base.text,
                    score=fused_score,
                    source_path=base.source_path,
                    heading=base.heading,
                    chunk_index=base.chunk_index,
                )
            )
        logger.debug(
            "Hybrid search: dense=%d sparse=%d fused=%d (top_k=%d)",
            len(dense_results),
            len(sparse_results),
            len(results),
            top_k,
        )
        return results
