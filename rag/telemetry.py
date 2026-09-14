"""OpenTelemetry tracing setup and instrumentation (Phase 5).

Two layers of tracing compose here:

1. **Automatic**: ``openinference-instrumentation-litellm`` patches
   ``litellm.completion``/``acompletion`` globally to emit a span per LLM
   call carrying model, prompts, responses, token counts, latency, and
   errors - no code changes needed in ``rag/generator.py`` or
   ``rag/query_rewriter.py``, both of which already call through litellm.
2. **Manual**: ``rag/pipeline.py`` wraps its retrieve/rerank/generate stages
   in spans created via :func:`get_tracer` here, so retrieval and reranking
   (which never go through litellm) are just as visible as generation is.

:func:`setup_telemetry` is safe to call from a request-serving process that
may have no reachable Phoenix collector (local dev without
``docker compose``, CI, unit tests importing ``app.main``): span export runs
on a background thread via ``BatchSpanProcessor`` and export failures are
logged, never raised - a query must never fail because Phoenix is down.
"""

from __future__ import annotations

import logging

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import Tracer

from config import Settings

logger = logging.getLogger("rag.telemetry")

_initialized = False


def setup_telemetry(settings: Settings) -> None:
    """Configure the global OTel tracer provider and instrument LiteLLM.

    Idempotent (a second call is a no-op) so it's safe to invoke from a
    FastAPI lifespan handler that may run more than once in the same
    process (e.g. multiple ``TestClient`` context managers in a test suite).
    """
    global _initialized
    if _initialized or not settings.otel_enabled:
        return

    try:
        resource = Resource.create({SERVICE_NAME: settings.otel_service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=settings.phoenix_collector_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        from openinference.instrumentation.litellm import LiteLLMInstrumentor

        LiteLLMInstrumentor().instrument(tracer_provider=provider)

        _initialized = True
        logger.info(
            "OpenTelemetry tracing initialized: service=%s collector=%s",
            settings.otel_service_name,
            settings.phoenix_collector_endpoint,
        )
    except Exception:  # noqa: BLE001 - tracing must never block the app from serving
        logger.exception("Failed to initialize OpenTelemetry tracing; continuing without it")


def get_tracer() -> Tracer:
    return trace.get_tracer("rag.pipeline")
