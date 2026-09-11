"""Unit tests for the FastAPI serving layer (app/main.py).

The real RagPipeline is never constructed here - ``get_pipeline`` is
overridden with a fake so these tests exercise routing, request/response
schema, and error mapping without touching Qdrant, the embedding model, or
an LLM provider.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.dependencies import get_pipeline
from app.main import app
from rag.generator import GenerationError
from rag.models import PipelineResult, RetrievedChunk, TokenUsage
from rag.stores.qdrant import VectorStoreError


def make_retrieved_chunk(chunk_id: str) -> RetrievedChunk:
    doc_id = chunk_id.split("#")[0]
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=f"text for {chunk_id}",
        score=0.9,
        source_path=f"data/corpus/{doc_id}.txt",
        heading="H",
        chunk_index=0,
    )


class FakePipeline:
    def __init__(
        self, result: PipelineResult | None = None, error: Exception | None = None
    ) -> None:
        self._result = result
        self._error = error
        self.received_question: str | None = None

    def query(self, question: str, trace_id: str | None = None) -> PipelineResult:
        self.received_question = question
        if self._error:
            raise self._error
        assert self._result is not None
        return self._result


def make_result(**overrides: object) -> PipelineResult:
    retrieved_chunk_ids = overrides.get(
        "retrieved_chunk_ids", ["ops-guide#14", "ops-guide#15", "ops-guide#83"]
    )
    defaults: dict[str, object] = {
        "answer": "The retention period is 90 days by default [ops-guide#14].",
        "citations": ["ops-guide#14"],
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "retrieved_chunks": [make_retrieved_chunk(cid) for cid in retrieved_chunk_ids],  # type: ignore[union-attr]
        "latency_ms": {"embed": 38.7, "retrieve": 12.1, "generate": 621.3, "total": 672.1},
        "usage": TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        "trace_id": "00000000-0000-0000-0000-000000000000",
    }
    defaults.update(overrides)
    return PipelineResult(**defaults)  # type: ignore[arg-type]


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_endpoint_returns_full_contract() -> None:
    fake = FakePipeline(result=make_result())
    app.dependency_overrides[get_pipeline] = lambda: fake
    try:
        with TestClient(app) as client:
            response = client.post("/query", json={"question": "What is the retention period?"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The retention period is 90 days by default [ops-guide#14]."
    assert body["citations"] == ["ops-guide#14"]
    assert body["retrieved_chunk_ids"] == ["ops-guide#14", "ops-guide#15", "ops-guide#83"]
    assert [c["chunk_id"] for c in body["retrieved_chunks"]] == [
        "ops-guide#14",
        "ops-guide#15",
        "ops-guide#83",
    ]
    assert body["retrieved_chunks"][0]["text"] == "text for ops-guide#14"
    assert body["latency_ms"] == {
        "embed": 38.7,
        "retrieve": 12.1,
        "generate": 621.3,
        "total": 672.1,
    }
    assert body["usage"] == {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    assert "trace_id" in body
    assert fake.received_question == "What is the retention period?"


def test_query_endpoint_preserves_retrieval_rank_order() -> None:
    fake = FakePipeline(result=make_result(retrieved_chunk_ids=["z#9", "a#0", "m#4"]))
    app.dependency_overrides[get_pipeline] = lambda: fake
    try:
        with TestClient(app) as client:
            response = client.post("/query", json={"question": "q"})
    finally:
        app.dependency_overrides.clear()

    assert response.json()["retrieved_chunk_ids"] == ["z#9", "a#0", "m#4"]


def test_query_endpoint_rejects_empty_question() -> None:
    fake = FakePipeline(result=make_result())
    app.dependency_overrides[get_pipeline] = lambda: fake
    try:
        with TestClient(app) as client:
            response = client.post("/query", json={"question": ""})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 422


def test_query_endpoint_maps_generation_error_to_502() -> None:
    fake = FakePipeline(error=GenerationError("LLM call failed"))
    app.dependency_overrides[get_pipeline] = lambda: fake
    try:
        with TestClient(app) as client:
            response = client.post("/query", json={"question": "q"})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 502


def test_query_endpoint_maps_vector_store_error_to_502() -> None:
    fake = FakePipeline(error=VectorStoreError("qdrant unreachable"))
    app.dependency_overrides[get_pipeline] = lambda: fake
    try:
        with TestClient(app) as client:
            response = client.post("/query", json={"question": "q"})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 502


def test_query_endpoint_maps_unexpected_error_to_500() -> None:
    fake = FakePipeline(error=ValueError("boom"))
    app.dependency_overrides[get_pipeline] = lambda: fake
    try:
        with TestClient(app) as client:
            response = client.post("/query", json={"question": "q"})
    finally:
        app.dependency_overrides.clear()
    assert response.status_code == 500
