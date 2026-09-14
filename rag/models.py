"""Shared data models used across chunking, embedding, retrieval, and generation.

These are the value types that flow between pipeline stages defined in
``rag.protocols``. Keeping them in one place means every stage agrees on the
same shapes without importing each other's implementation modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass(frozen=True)
class Chunk:
    """A chunk of a source document, ready to be embedded and stored."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    text: str
    source_path: str
    heading: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "chunk_index": self.chunk_index,
            "source_path": self.source_path,
            "heading": self.heading,
            "chunk_id": self.chunk_id,
        }


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by vector search, with its similarity score."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    source_path: str
    heading: str
    chunk_index: int


class TokenUsage(BaseModel):
    """Token accounting for a single LLM generation call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class GenerationResult:
    """The generator's output: a grounded answer plus verified citations."""

    answer: str
    citations: list[str]
    usage: TokenUsage
    cost_usd: float = 0.0


@dataclass(frozen=True)
class PipelineResult:
    """The full result of one end-to-end pipeline run, ready for API serialization."""

    answer: str
    citations: list[str]
    retrieved_chunk_ids: list[str]
    retrieved_chunks: list[RetrievedChunk]
    latency_ms: dict[str, float]
    usage: TokenUsage
    cost_usd: float
    trace_id: str
