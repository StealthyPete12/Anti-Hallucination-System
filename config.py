"""Centralized, environment-driven configuration for the RAG system.

Every tunable used by ``rag/`` and ``app/`` lives here. Override any field
via environment variables prefixed ``RAG_`` (e.g. ``RAG_TOP_K=8``) or a
``.env`` file in the project root - see ``.env.example``.
"""

from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_file=".env", extra="ignore")

    # --- Corpus / ingestion -------------------------------------------------
    corpus_dir: str = "data/corpus"

    # --- Chunking -------------------------------------------------------------
    chunk_size_tokens: int = 512
    chunk_overlap_ratio: float = 0.15

    # --- Embedding --------------------------------------------------------
    embedding_model_name: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: str = ".cache/embeddings"
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"

    # --- Vector store (Qdrant) ---------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_name: str = "postgres_docs"

    # --- Retrieval ----------------------------------------------------------
    top_k: int = 5

    # --- Phase 4: hybrid retrieval --------------------------------------------
    # "sparse"/"hybrid" require a vector store that supports named dense+sparse
    # vectors (see rag.stores.qdrant). Invalid values fail at Settings
    # construction time, before any pipeline/experiment is built.
    retrieval_mode: Literal["dense", "sparse", "hybrid"] = "dense"
    rrf_k: int = 60  # reciprocal-rank-fusion rank discount constant
    # Candidates fetched per retrieval branch (dense/sparse) before fusion
    # and/or reranking narrows down to top_k.
    retrieval_fanout: int = 20

    # --- Phase 4: cross-encoder reranking --------------------------------------
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_device: str = "cpu"

    # --- Phase 4: query rewriting ----------------------------------------------
    query_rewrite_enabled: bool = False
    query_rewrite_model: str = "gpt-4o-mini"
    query_rewrite_timeout_seconds: float = 30.0

    # --- Generation (LiteLLM) ------------------------------------------------
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_timeout_seconds: float = 60.0

    # --- Evaluation judge (LiteLLM, Phase 2) ---------------------------------
    judge_model: str = "gpt-4o-mini"
    judge_timeout_seconds: float = 60.0
    judge_max_retries: int = 3

    # --- API ------------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Logging --------------------------------------------------------------
    log_level: str = "INFO"

    # --- Phase 5: OpenTelemetry / Phoenix tracing ------------------------------
    # Off by default (same convention as reranker_enabled/query_rewrite_enabled
    # above) - a unit test importing app.main shouldn't spend seconds retrying
    # span export against a Phoenix collector that isn't running. docker-compose
    # turns this on for the api service, where Phoenix is always available.
    otel_enabled: bool = False
    otel_service_name: str = "anti-hallucination-rag"
    # Arize Phoenix's OTLP HTTP collector (see docker-compose.yml's `phoenix`
    # service). Unreachable-endpoint export failures are logged, never raised -
    # the API must keep serving queries even if Phoenix is down.
    phoenix_collector_endpoint: str = "http://localhost:6006/v1/traces"

    # --- Phase 5: rolling-window operational metrics ---------------------------
    # In-process only (see rag/metrics.py) - resets on restart, not shared across
    # replicas. Phoenix/Grafana are the durable, cross-replica source of truth;
    # this is a cheap, dependency-free "what's happening right now" view.
    metrics_window_size: int = 100

    # --- Phase 5: Postgres (feedback persistence) ------------------------------
    postgres_dsn: str = "postgresql+psycopg://rag:rag@localhost:5432/rag_observability"


#: Per-1K-token pricing used as the source of truth for request cost
#: calculation (see rag/cost.py). Deliberately separate from `Settings` -
#: this is a static reference table, not an environment-driven tunable, and
#: keeping it here (module-level, like eval/compare.py's GATE) makes every
#: known model's price a one-line diff instead of a new env var.
#:
#: A model not listed here isn't an error: rag/cost.py falls back to
#: litellm's own (usually more current) pricing table.
#:
#: Rates as of Phase 5 authoring (2026-09); update as provider pricing changes.
MODEL_RATES: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input_per_1k": 0.00015, "output_per_1k": 0.0006},
    "gpt-4o": {"input_per_1k": 0.0025, "output_per_1k": 0.01},
    "gpt-4.1-mini": {"input_per_1k": 0.0004, "output_per_1k": 0.0016},
    "claude-sonnet-5": {"input_per_1k": 0.003, "output_per_1k": 0.015},
    "claude-haiku-4-5-20251001": {"input_per_1k": 0.001, "output_per_1k": 0.005},
}


settings = Settings()
