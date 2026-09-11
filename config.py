"""Centralized, environment-driven configuration for the RAG system.

Every tunable used by ``rag/`` and ``app/`` lives here. Override any field
via environment variables prefixed ``RAG_`` (e.g. ``RAG_TOP_K=8``) or a
``.env`` file in the project root - see ``.env.example``.
"""

from __future__ import annotations

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

    # --- Generation (LiteLLM) ------------------------------------------------
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024
    llm_timeout_seconds: float = 60.0

    # --- API ------------------------------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Logging --------------------------------------------------------------
    log_level: str = "INFO"


settings = Settings()
