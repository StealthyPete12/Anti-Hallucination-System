"""Qdrant-backed vector store for chunk embeddings.

Point IDs are deterministic (uuid5 of the chunk_id), so re-indexing
unchanged chunks overwrites the same points instead of duplicating them.
Full chunk metadata is stored in the payload alongside the vector so it can
be returned directly from search and filtered on in future phases (e.g.
restricting retrieval to a doc_id or heading).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from rag.models import Chunk, RetrievedChunk

logger = logging.getLogger("rag.stores.qdrant")


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
    """Dense vector store backed by a Qdrant collection."""

    def __init__(self, url: str, collection_name: str, timeout: float = 30.0) -> None:
        self.url = url
        self.collection_name = collection_name
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
        logger.info("Creating collection %s (dim=%d)", name, vector_size)
        try:
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
                vector=vector,
                payload={"text": chunk.text, **chunk.to_metadata()},
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        try:
            self._client.upsert(collection_name=self.collection_name, points=points)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to upsert {len(points)} point(s): {exc}") from exc
        logger.info("Upserted %d point(s) into %s", len(points), self.collection_name)

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        qfilter = _build_filter(filters)
        try:
            response = self._client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=qfilter,
                with_payload=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Search failed: {exc}") from exc

        retrieved = [
            RetrievedChunk(
                chunk_id=point.payload["chunk_id"],
                doc_id=point.payload["doc_id"],
                text=point.payload["text"],
                score=point.score,
                source_path=point.payload.get("source_path", ""),
                heading=point.payload.get("heading", ""),
                chunk_index=point.payload.get("chunk_index", -1),
            )
            for point in response.points
            if point.payload is not None
        ]
        logger.debug("Search returned %d result(s)", len(retrieved))
        return retrieved
