"""Application dependency wiring: builds the RagPipeline singleton from Settings.

Kept separate from ``app.main`` so FastAPI's ``Depends`` mechanism can
override ``get_pipeline`` in tests with a fake pipeline, without touching
any real model, Qdrant, or LLM calls.

``build_pipeline(settings)`` is a pure factory (no caching, no global
state) so Phase 4's experiment framework (``eval/experiment.py``) can build
many differently-configured pipelines - different embedding models, chunk
sizes, retrieval modes, rerankers - in the same process without any of them
colliding through ``get_pipeline``'s ``lru_cache``.
"""

from __future__ import annotations

from functools import lru_cache

from config import Settings, settings
from rag.chunking import ChunkingConfig, RecursiveCharacterChunker
from rag.embedders.local import LocalBGEEmbedder
from rag.generator import LiteLLMGenerator
from rag.pipeline import RagPipeline
from rag.query_rewriter import LiteLLMQueryRewriter
from rag.reranker import CrossEncoderReranker
from rag.sparse import BM25Vectorizer
from rag.stores.qdrant import QdrantVectorStore


def build_embedder(settings: Settings) -> LocalBGEEmbedder:
    return LocalBGEEmbedder(
        model_name=settings.embedding_model_name,
        cache_dir=settings.embedding_cache_dir,
        batch_size=settings.embedding_batch_size,
        device=settings.embedding_device,
    )


def build_pipeline(settings: Settings) -> RagPipeline:
    """Build a fully-wired :class:`RagPipeline` from a ``Settings`` instance.

    Pure: every call builds fresh components from the given settings, with
    no caching and no shared global state - safe to call repeatedly with
    different settings (e.g. one call per Phase 4 experiment configuration).
    """
    embedder = build_embedder(settings)
    chunker = RecursiveCharacterChunker(
        config=ChunkingConfig(
            chunk_size_tokens=settings.chunk_size_tokens,
            overlap_ratio=settings.chunk_overlap_ratio,
        ),
        token_counter=embedder.count_tokens,
    )
    vector_store = QdrantVectorStore(
        url=settings.qdrant_url,
        collection_name=settings.qdrant_collection_name,
        enable_sparse=settings.retrieval_mode in ("sparse", "hybrid"),
    )
    generator = LiteLLMGenerator(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout_seconds,
    )

    sparse_encoder = BM25Vectorizer() if settings.retrieval_mode in ("sparse", "hybrid") else None
    reranker = (
        CrossEncoderReranker(model_name=settings.reranker_model, device=settings.reranker_device)
        if settings.reranker_enabled
        else None
    )
    query_rewriter = (
        LiteLLMQueryRewriter(
            model=settings.query_rewrite_model, timeout=settings.query_rewrite_timeout_seconds
        )
        if settings.query_rewrite_enabled
        else None
    )

    return RagPipeline(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        generator=generator,
        top_k=settings.top_k,
        retrieval_mode=settings.retrieval_mode,
        sparse_encoder=sparse_encoder,
        reranker=reranker,
        retrieval_fanout=settings.retrieval_fanout,
        rrf_k=settings.rrf_k,
        query_rewriter=query_rewriter,
    )


@lru_cache(maxsize=1)
def get_embedder() -> LocalBGEEmbedder:
    return build_embedder(settings)


@lru_cache(maxsize=1)
def get_pipeline() -> RagPipeline:
    return build_pipeline(settings)
