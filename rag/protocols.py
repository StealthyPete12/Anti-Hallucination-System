"""Protocol contracts for RAG pipeline components.

These Protocols are the architectural seams between pipeline stages: any
class implementing the right methods satisfies the contract structurally, no
inheritance required. Future phases (hybrid retrieval, rerankers, query
rewriting, agents, graph RAG, ...) swap implementations behind these same
interfaces without touching ``rag.pipeline`` or each other.

No business logic lives here — method signatures only.
"""

from __future__ import annotations

from typing import Any, Protocol

from rag.models import Chunk, GenerationResult, RetrievedChunk


class Chunker(Protocol):
    """Splits a source document into stably-identified chunks with metadata."""

    def chunk_document(self, *, doc_id: str, text: str, source_path: str) -> list[Chunk]: ...


class Embedder(Protocol):
    """Produces dense vector embeddings for documents and queries."""

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class VectorStore(Protocol):
    """Persists chunk embeddings and metadata, and serves dense vector search."""

    def create_collection(self, name: str | None = None, vector_size: int = 0) -> None: ...

    def collection_exists(self, name: str | None = None) -> bool: ...

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(
        self,
        query_vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]: ...

    def delete_collection(self, name: str | None = None) -> None: ...


class Generator(Protocol):
    """Turns a query and its retrieved chunks into a grounded, cited answer."""

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> GenerationResult: ...
