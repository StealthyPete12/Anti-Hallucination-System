"""Application dependency wiring: builds the RagPipeline singleton from Settings.

Kept separate from ``app.main`` so FastAPI's ``Depends`` mechanism can
override ``get_pipeline`` in tests with a fake pipeline, without touching
any real model, Qdrant, or LLM calls.
"""

from __future__ import annotations

from functools import lru_cache

from config import settings
from rag.chunking import ChunkingConfig, RecursiveCharacterChunker
from rag.embedders.local import LocalBGEEmbedder
from rag.generator import LiteLLMGenerator
from rag.pipeline import RagPipeline
from rag.stores.qdrant import QdrantVectorStore


@lru_cache(maxsize=1)
def get_embedder() -> LocalBGEEmbedder:
    return LocalBGEEmbedder(
        model_name=settings.embedding_model_name,
        cache_dir=settings.embedding_cache_dir,
        batch_size=settings.embedding_batch_size,
        device=settings.embedding_device,
    )


@lru_cache(maxsize=1)
def get_pipeline() -> RagPipeline:
    embedder = get_embedder()
    chunker = RecursiveCharacterChunker(
        config=ChunkingConfig(
            chunk_size_tokens=settings.chunk_size_tokens,
            overlap_ratio=settings.chunk_overlap_ratio,
        ),
        token_counter=embedder.count_tokens,
    )
    vector_store = QdrantVectorStore(
        url=settings.qdrant_url, collection_name=settings.qdrant_collection_name
    )
    generator = LiteLLMGenerator(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
    )
    return RagPipeline(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        generator=generator,
        top_k=settings.top_k,
    )
