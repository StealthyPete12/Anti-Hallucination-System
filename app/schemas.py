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
    cost_usd: float
    trace_id: str
    # Phase 5: runtime citation validation (rag.citation_validator). Default
    # to "passed, nothing invalid" so any pre-Phase-5 caller constructing a
    # QueryResponse without these stays valid.
    citation_validation_passed: bool = True
    invalid_citations: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str = "ok"


class FeedbackRequest(BaseModel):
    """Correlates a user rating back to the trace_id of the request it's
    about - see app/db/models.py::Feedback and the README's Phase 5 section
    on trace correlation."""

    trace_id: str = Field(..., min_length=1)
    rating: int = Field(..., ge=-1, le=1, description="Thumbs down (-1), neutral (0), up (1)")
    comment: str | None = None


class FeedbackResponse(BaseModel):
    success: bool = True


class MetricsResponse(BaseModel):
    count: int
    mean_cost_usd: float | None
    p95_latency_ms: float | None
    window_size: int
