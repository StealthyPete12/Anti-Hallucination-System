"""Protocol contracts for RAG pipeline components.

These Protocols are the architectural seams between pipeline stages: any
class implementing the right methods satisfies the contract structurally, no
inheritance required. Phase 4 (hybrid retrieval, rerankers, query
rewriting) and future phases (agents, graph RAG, ...) swap implementations
behind these same interfaces without touching ``rag.pipeline`` or each other.

No business logic lives here — method signatures only.
"""

from __future__ import annotations

from typing import Any, Protocol

from rag.models import Chunk, GenerationResult, RetrievedChunk, SparseVector


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


class HybridVectorStore(VectorStore, Protocol):
    """A VectorStore that additionally supports sparse and fused dense+sparse
    search (Phase 4). Only ``retrieval_mode in {"sparse", "hybrid"}`` requires
    an implementation of this - dense-only fakes never need it."""

    def upsert_sparse(self, chunks: list[Chunk], sparse_vectors: list[SparseVector]) -> None: ...

    def search_sparse(
        self,
        sparse_query: SparseVector,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]: ...

    def search_hybrid(
        self,
        query_vector: list[float],
        sparse_query: SparseVector,
        top_k: int,
        *,
        fanout: int,
        rrf_k: int = 60,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]: ...


class Generator(Protocol):
    """Turns a query and its retrieved chunks into a grounded, cited answer."""

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> GenerationResult: ...


class SparseEncoder(Protocol):
    """Encodes text into BM25 sparse vectors for sparse/hybrid retrieval (Phase 4)."""

    def encode_document(self, text: str) -> SparseVector: ...

    def encode_query(self, text: str) -> SparseVector: ...


class Reranker(Protocol):
    """Re-scores an initial candidate set with a cross-encoder and returns the top N (Phase 4)."""

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]: ...


class QueryRewriter(Protocol):
    """Rewrites a user question into a retrieval-optimized query (Phase 4)."""

    def rewrite(self, question: str) -> str: ...
