"""End-to-end integration test: real chunker + real embedder + real Qdrant.

Downloads BAAI/bge-small-en-v1.5 (cached after the first run) and embeds
against an in-memory Qdrant instance. No LLM call is made - the generator is
still faked, since that requires a paid provider API key which isn't
available in CI. This test is the one place that proves chunking, the real
BGE model (including the query-prefix asymmetry), and Qdrant search actually
agree with each other end to end.

Skipped automatically if the model can't be downloaded (e.g. no network).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.chunking import ChunkingConfig, RecursiveCharacterChunker
from rag.embedders.local import EmbeddingError, LocalBGEEmbedder
from rag.generator import GenerationResult
from rag.models import RetrievedChunk, TokenUsage
from rag.pipeline import RagPipeline
from rag.stores.qdrant import QdrantVectorStore

pytestmark = pytest.mark.integration


class StubGenerator:
    def generate(self, query: str, chunks: list[RetrievedChunk]) -> GenerationResult:
        citations = [c.chunk_id for c in chunks[:1]]
        return GenerationResult(
            answer=f"stub answer citing {citations}", citations=citations, usage=TokenUsage()
        )


@pytest.fixture(scope="module")
def embedder() -> LocalBGEEmbedder:
    embedder = LocalBGEEmbedder()
    try:
        _ = embedder.dimension  # forces model load
    except EmbeddingError as exc:
        pytest.skip(f"Could not load embedding model (offline?): {exc}")
    return embedder


def test_index_and_query_real_embeddings(embedder: LocalBGEEmbedder, tmp_path: Path) -> None:
    corpus_dir = tmp_path
    (corpus_dir / "postgres_mvcc.txt").write_text(
        "MVCC stands for Multiversion Concurrency Control. It lets readers "
        "and writers avoid blocking each other in PostgreSQL.",
        encoding="utf-8",
    )
    (corpus_dir / "postgres_indexes.txt").write_text(
        "By default, CREATE INDEX builds a B-tree index, which handles "
        "equality and range queries efficiently.",
        encoding="utf-8",
    )

    chunker = RecursiveCharacterChunker(
        config=ChunkingConfig(chunk_size_tokens=64, overlap_ratio=0.15),
        token_counter=embedder.count_tokens,
    )
    vector_store = QdrantVectorStore(url=":memory:", collection_name="e2e_test")
    pipeline = RagPipeline(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        generator=StubGenerator(),
        top_k=2,
    )

    indexed = pipeline.index_corpus(corpus_dir)
    assert indexed == 2

    result = pipeline.query("What does MVCC stand for?")

    assert result.retrieved_chunk_ids
    assert result.retrieved_chunk_ids[0].startswith("postgres_mvcc")
    assert result.citations == [result.retrieved_chunk_ids[0]]
    assert set(result.latency_ms) == {"embed", "retrieve", "generate", "total"}
