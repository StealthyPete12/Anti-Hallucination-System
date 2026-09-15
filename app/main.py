"""FastAPI application: GET /health and POST /query.

/query returns exactly the fields the evaluation harness needs
(``retrieved_chunk_ids``, ``latency_ms``, ``usage``, ``trace_id``) from a
single production code path - there is no separate "evaluation" pipeline to
keep in sync.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.repository import FeedbackRepository, RequestMetricRepository
from app.db.session import build_engine, get_db, init_db
from app.dependencies import get_pipeline
from app.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
    RetrievedChunkOut,
)
from config import settings
from rag.generator import GenerationError
from rag.metrics import metrics_recorder
from rag.pipeline import RagPipeline
from rag.stores.qdrant import VectorStoreError
from rag.telemetry import setup_telemetry

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_telemetry(settings)  # never raises - see rag/telemetry.py
    try:
        init_db(build_engine(settings))
    except SQLAlchemyError:
        logger.warning(
            "Could not reach Postgres at startup; /feedback will fail until it's reachable",
            exc_info=True,
        )
    yield


app = FastAPI(title="Anti-Hallucination RAG API", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/metrics", response_model=MetricsResponse)
def metrics() -> MetricsResponse:
    """Rolling-window operational metrics over the last
    ``RAG_METRICS_WINDOW_SIZE`` requests served by this process (see
    rag/metrics.py). Phoenix is the durable, cross-replica source of truth;
    this is a cheap in-process supplement."""
    snapshot = metrics_recorder.snapshot()
    return MetricsResponse(
        count=snapshot.count,
        mean_cost_usd=snapshot.mean_cost_usd,
        p95_latency_ms=snapshot.p95_latency_ms,
        window_size=snapshot.window_size,
    )


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest, db: Session = Depends(get_db)) -> FeedbackResponse:  # noqa: B008
    """Persist a user rating, correlated to the trace_id of the request it's
    about (see app/db/models.py::Feedback for why that column matters)."""
    try:
        FeedbackRepository(db).create(
            trace_id=request.trace_id, rating=request.rating, comment=request.comment
        )
    except SQLAlchemyError as exc:
        logger.error("Could not persist feedback for trace_id=%s: %s", request.trace_id, exc)
        raise HTTPException(status_code=503, detail="Feedback storage unavailable") from exc
    return FeedbackResponse(success=True)


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    pipeline: RagPipeline = Depends(get_pipeline),  # noqa: B008 - standard FastAPI DI pattern
    db: Session = Depends(get_db),  # noqa: B008
) -> QueryResponse:
    trace_id = str(uuid.uuid4())
    logger.info("trace_id=%s Request received: question=%r", trace_id, request.question)

    try:
        result = pipeline.query(request.question, trace_id=trace_id)
    except (GenerationError, VectorStoreError) as exc:
        logger.error("trace_id=%s Query failed: %s", trace_id, exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("trace_id=%s Unexpected failure", trace_id)
        raise HTTPException(status_code=500, detail="Internal error") from exc

    try:
        RequestMetricRepository(db).create(
            trace_id=result.trace_id,
            cost_usd=result.cost_usd,
            latency_ms=result.latency_ms["total"],
            retrieval_mode=settings.retrieval_mode,
            citation_validation_passed=result.citation_validation_passed,
        )
    except SQLAlchemyError:
        # Postgres being down must never fail a query that already succeeded -
        # Phoenix and rag/metrics.py's in-process window still have this
        # request's cost/latency even if this row never lands.
        logger.warning("trace_id=%s Could not persist request metric", trace_id, exc_info=True)

    return QueryResponse(
        answer=result.answer,
        citations=result.citations,
        retrieved_chunk_ids=result.retrieved_chunk_ids,
        retrieved_chunks=[
            RetrievedChunkOut(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                score=chunk.score,
                heading=chunk.heading,
                chunk_index=chunk.chunk_index,
            )
            for chunk in result.retrieved_chunks
        ],
        latency_ms=result.latency_ms,
        usage=result.usage,
        cost_usd=result.cost_usd,
        trace_id=result.trace_id,
        citation_validation_passed=result.citation_validation_passed,
        invalid_citations=result.invalid_citations,
    )
