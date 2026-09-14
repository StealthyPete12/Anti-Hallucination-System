"""FastAPI application: GET /health and POST /query.

/query returns exactly the fields the evaluation harness needs
(``retrieved_chunk_ids``, ``latency_ms``, ``usage``, ``trace_id``) from a
single production code path - there is no separate "evaluation" pipeline to
keep in sync.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import Depends, FastAPI, HTTPException

from app.dependencies import get_pipeline
from app.schemas import HealthResponse, QueryRequest, QueryResponse, RetrievedChunkOut
from config import settings
from rag.generator import GenerationError
from rag.pipeline import RagPipeline
from rag.stores.qdrant import VectorStoreError

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("app")

app = FastAPI(title="Anti-Hallucination RAG API", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    pipeline: RagPipeline = Depends(get_pipeline),  # noqa: B008 - standard FastAPI DI pattern
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
    )
