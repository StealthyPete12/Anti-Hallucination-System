"""Request/response schemas for the FastAPI query endpoint."""

from __future__ import annotations

from pydantic import BaseModel, Field

from rag.models import TokenUsage


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)


class RetrievedChunkOut(BaseModel):
    """A retrieved chunk with its text, so callers (e.g. the eval harness) don't
    have to re-derive chunk boundaries or re-chunk the corpus themselves to see
    what the generator actually saw."""

    chunk_id: str
    doc_id: str
    text: str
    score: float
    heading: str
    chunk_index: int


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    retrieved_chunk_ids: list[str]
    retrieved_chunks: list[RetrievedChunkOut]
    latency_ms: dict[str, float]
    usage: TokenUsage
    trace_id: str


class HealthResponse(BaseModel):
    status: str = "ok"
