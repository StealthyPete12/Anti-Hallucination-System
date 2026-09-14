"""Unit tests for the pure build_pipeline(settings) factory (app/dependencies.py).

No model weights are downloaded and no real Qdrant server is needed:
LocalBGEEmbedder lazy-loads its model only on first use, and
QdrantClient(location=...) doesn't touch the network until a real request is
made - so these tests only check that Settings correctly drive *which*
components get wired together for Phase 4's config-driven retrieval modes.
"""

from __future__ import annotations

from app.dependencies import build_pipeline
from config import Settings
from rag.query_rewriter import LiteLLMQueryRewriter
from rag.reranker import CrossEncoderReranker
from rag.sparse import BM25Vectorizer


def test_default_settings_build_dense_pipeline_with_no_phase4_extras() -> None:
    pipeline = build_pipeline(Settings(qdrant_url=":memory:"))
    assert pipeline.retrieval_mode == "dense"
    assert pipeline.sparse_encoder is None
    assert pipeline.reranker is None
    assert pipeline.query_rewriter is None
    assert pipeline.vector_store.enable_sparse is False  # type: ignore[attr-defined]


def test_sparse_mode_wires_sparse_encoder_and_enables_store_sparse_vectors() -> None:
    pipeline = build_pipeline(Settings(qdrant_url=":memory:", retrieval_mode="sparse"))
    assert pipeline.retrieval_mode == "sparse"
    assert isinstance(pipeline.sparse_encoder, BM25Vectorizer)
    assert pipeline.vector_store.enable_sparse is True  # type: ignore[attr-defined]


def test_hybrid_mode_wires_sparse_encoder_too() -> None:
    pipeline = build_pipeline(Settings(qdrant_url=":memory:", retrieval_mode="hybrid"))
    assert pipeline.retrieval_mode == "hybrid"
    assert isinstance(pipeline.sparse_encoder, BM25Vectorizer)


def test_reranker_enabled_wires_configured_model() -> None:
    pipeline = build_pipeline(
        Settings(
            qdrant_url=":memory:", reranker_enabled=True, reranker_model="BAAI/bge-reranker-base"
        )
    )
    assert isinstance(pipeline.reranker, CrossEncoderReranker)
    assert pipeline.reranker.model_name == "BAAI/bge-reranker-base"


def test_reranker_disabled_by_default() -> None:
    pipeline = build_pipeline(Settings(qdrant_url=":memory:"))
    assert pipeline.reranker is None


def test_query_rewrite_enabled_wires_rewriter() -> None:
    pipeline = build_pipeline(
        Settings(qdrant_url=":memory:", query_rewrite_enabled=True, query_rewrite_model="gpt-4o")
    )
    assert isinstance(pipeline.query_rewriter, LiteLLMQueryRewriter)
    assert pipeline.query_rewriter.model == "gpt-4o"


def test_settings_top_k_and_fanout_propagate() -> None:
    pipeline = build_pipeline(Settings(qdrant_url=":memory:", top_k=7, retrieval_fanout=25))
    assert pipeline.top_k == 7
    assert pipeline.retrieval_fanout == 25


def test_settings_chunking_config_propagates() -> None:
    pipeline = build_pipeline(
        Settings(qdrant_url=":memory:", chunk_size_tokens=1024, chunk_overlap_ratio=0.2)
    )
    assert pipeline.chunker.config.chunk_size_tokens == 1024
    assert pipeline.chunker.config.overlap_ratio == 0.2


def test_each_call_builds_independent_components() -> None:
    settings_a = Settings(qdrant_url=":memory:", top_k=3)
    settings_b = Settings(qdrant_url=":memory:", top_k=9)
    pipeline_a = build_pipeline(settings_a)
    pipeline_b = build_pipeline(settings_b)
    assert pipeline_a.top_k == 3
    assert pipeline_b.top_k == 9
    assert pipeline_a.vector_store is not pipeline_b.vector_store
