"""Unit tests for RagPipeline orchestration (rag/pipeline.py).

Every collaborator (Chunker, Embedder, VectorStore, Generator) is a small
fake satisfying the Protocol in rag/protocols.py - dependency injection
means the pipeline can be tested end to end with no model weights, no
running Qdrant, and no LLM API calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from rag.models import Chunk, GenerationResult, RetrievedChunk, SparseVector, TokenUsage
from rag.pipeline import PipelineConfigError, RagPipeline


class FakeChunker:
    def chunk_document(self, *, doc_id: str, text: str, source_path: str) -> list[Chunk]:
        return [
            Chunk(
                chunk_id=f"{doc_id}#0",
                doc_id=doc_id,
                chunk_index=0,
                text=text,
                source_path=source_path,
                heading="H",
            )
        ]


class FakeEmbedder:
    dimension = 4

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


class FakeVectorStore:
    def __init__(self, search_results: list[RetrievedChunk] | None = None) -> None:
        self._exists = False
        self.created_with_dim: int | None = None
        self.upserted: list[tuple[list[Chunk], list[list[float]]]] = []
        self._search_results = search_results or []

    def collection_exists(self, name: str | None = None) -> bool:
        return self._exists

    def create_collection(self, name: str | None = None, vector_size: int = 0) -> None:
        self.created_with_dim = vector_size
        self._exists = True

    def delete_collection(self, name: str | None = None) -> None:
        self._exists = False

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self.upserted.append((chunks, vectors))

    def search(
        self, query_vector: list[float], top_k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        return self._search_results[:top_k]


class FakeGenerator:
    def __init__(self, result: GenerationResult | None = None) -> None:
        self._result = result or GenerationResult(
            answer="fake answer", citations=[], usage=TokenUsage()
        )
        self.received_chunks: list[RetrievedChunk] | None = None

    def generate(self, query: str, chunks: list[RetrievedChunk]) -> GenerationResult:
        self.received_chunks = chunks
        return self._result


def make_retrieved(chunk_id: str) -> RetrievedChunk:
    doc_id = chunk_id.split("#")[0]
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text="text",
        score=0.5,
        source_path="x.txt",
        heading="H",
        chunk_index=0,
    )


def test_index_corpus_chunks_embeds_and_upserts(tmp_path: Path) -> None:
    (tmp_path / "doc1.txt").write_text("hello world", encoding="utf-8")
    (tmp_path / "doc2.txt").write_text("another document", encoding="utf-8")

    vector_store = FakeVectorStore()
    pipeline = RagPipeline(
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
        vector_store=vector_store,
        generator=FakeGenerator(),
    )

    total = pipeline.index_corpus(tmp_path)

    assert total == 2
    assert vector_store.created_with_dim == 4
    assert len(vector_store.upserted) == 2


def test_index_corpus_skips_collection_creation_if_exists(tmp_path: Path) -> None:
    (tmp_path / "doc1.txt").write_text("hello world", encoding="utf-8")
    vector_store = FakeVectorStore()
    vector_store._exists = True
    pipeline = RagPipeline(
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
        vector_store=vector_store,
        generator=FakeGenerator(),
    )
    pipeline.index_corpus(tmp_path)
    assert vector_store.created_with_dim is None


def test_index_corpus_empty_dir_returns_zero(tmp_path: Path) -> None:
    vector_store = FakeVectorStore()
    pipeline = RagPipeline(
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
        vector_store=vector_store,
        generator=FakeGenerator(),
    )
    assert pipeline.index_corpus(tmp_path) == 0
    assert vector_store.created_with_dim is None


def test_query_returns_retrieved_chunk_ids_in_rank_order() -> None:
    retrieved = [make_retrieved("a#0"), make_retrieved("a#1"), make_retrieved("b#3")]
    vector_store = FakeVectorStore(search_results=retrieved)
    pipeline = RagPipeline(
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
        vector_store=vector_store,
        generator=FakeGenerator(),
        top_k=3,
    )
    result = pipeline.query("some question")
    assert result.retrieved_chunk_ids == ["a#0", "a#1", "b#3"]
    assert result.retrieved_chunks == retrieved


def test_query_passes_retrieved_chunks_to_generator() -> None:
    retrieved = [make_retrieved("a#0")]
    vector_store = FakeVectorStore(search_results=retrieved)
    generator = FakeGenerator()
    pipeline = RagPipeline(
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
        vector_store=vector_store,
        generator=generator,
    )
    pipeline.query("some question")
    assert generator.received_chunks == retrieved


def test_query_result_has_all_required_fields() -> None:
    vector_store = FakeVectorStore(search_results=[make_retrieved("a#0")])
    generator = FakeGenerator(
        result=GenerationResult(
            answer="The answer is X [a#0].",
            citations=["a#0"],
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
    )
    pipeline = RagPipeline(
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
        vector_store=vector_store,
        generator=generator,
    )
    result = pipeline.query("q", trace_id="fixed-trace-id")

    assert result.answer == "The answer is X [a#0]."
    assert result.citations == ["a#0"]
    assert result.retrieved_chunk_ids == ["a#0"]
    assert result.retrieved_chunks == [make_retrieved("a#0")]
    assert set(result.latency_ms.keys()) == {"embed", "retrieve", "generate", "total"}
    assert all(isinstance(v, float) for v in result.latency_ms.values())
    assert result.usage.total_tokens == 15
    assert result.trace_id == "fixed-trace-id"


def test_query_generates_trace_id_when_not_provided() -> None:
    vector_store = FakeVectorStore(search_results=[])
    pipeline = RagPipeline(
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
        vector_store=vector_store,
        generator=FakeGenerator(),
    )
    result = pipeline.query("q")
    assert result.trace_id
    assert len(result.trace_id) == 36  # uuid4 string length


def test_query_respects_top_k() -> None:
    retrieved = [make_retrieved(f"a#{i}") for i in range(10)]
    vector_store = FakeVectorStore(search_results=retrieved)
    pipeline = RagPipeline(
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
        vector_store=vector_store,
        generator=FakeGenerator(),
        top_k=3,
    )
    result = pipeline.query("q")
    assert len(result.retrieved_chunk_ids) == 3


# --- Phase 4: hybrid retrieval, reranking, query rewriting -----------------


class FakeSparseEncoder:
    def __init__(self) -> None:
        self.fit_calls: list[list[str]] = []

    def fit(self, documents: list[str]) -> None:
        self.fit_calls.append(documents)

    def encode_document(self, text: str) -> SparseVector:
        return SparseVector(indices=[0], values=[1.0])

    def encode_query(self, text: str) -> SparseVector:
        return SparseVector(indices=[0], values=[1.0])


class FakeHybridVectorStore(FakeVectorStore):
    """A FakeVectorStore that also satisfies HybridVectorStore (sparse + hybrid search)."""

    def __init__(
        self,
        dense_results: list[RetrievedChunk] | None = None,
        sparse_results: list[RetrievedChunk] | None = None,
        hybrid_results: list[RetrievedChunk] | None = None,
    ) -> None:
        super().__init__(search_results=dense_results or [])
        self._sparse_results = sparse_results or []
        self._hybrid_results = hybrid_results or []
        self.sparse_upserted: list[tuple[list[Chunk], list[SparseVector]]] = []
        self.last_sparse_query: SparseVector | None = None

    def upsert_sparse(self, chunks: list[Chunk], sparse_vectors: list[SparseVector]) -> None:
        self.sparse_upserted.append((chunks, sparse_vectors))

    def search_sparse(
        self, sparse_query: SparseVector, top_k: int, filters: dict[str, Any] | None = None
    ) -> list[RetrievedChunk]:
        self.last_sparse_query = sparse_query
        return self._sparse_results[:top_k]

    def search_hybrid(
        self,
        query_vector: list[float],
        sparse_query: SparseVector,
        top_k: int,
        *,
        fanout: int,
        rrf_k: int = 60,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        self.last_sparse_query = sparse_query
        return self._hybrid_results[:top_k]


def test_invalid_retrieval_mode_raises() -> None:
    with pytest.raises(PipelineConfigError):
        RagPipeline(
            chunker=FakeChunker(),
            embedder=FakeEmbedder(),
            vector_store=FakeVectorStore(),
            generator=FakeGenerator(),
            retrieval_mode="nonsense",
        )


def test_sparse_mode_without_sparse_encoder_raises() -> None:
    with pytest.raises(PipelineConfigError):
        RagPipeline(
            chunker=FakeChunker(),
            embedder=FakeEmbedder(),
            vector_store=FakeVectorStore(),
            generator=FakeGenerator(),
            retrieval_mode="sparse",
        )


def test_sparse_mode_uses_search_sparse() -> None:
    sparse_result = [make_retrieved("s#0")]
    vector_store = FakeHybridVectorStore(sparse_results=sparse_result)
    pipeline = RagPipeline(
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
        vector_store=vector_store,
        generator=FakeGenerator(),
        retrieval_mode="sparse",
        sparse_encoder=FakeSparseEncoder(),
        top_k=5,
    )
    result = pipeline.query("q")
    assert result.retrieved_chunk_ids == ["s#0"]


def test_hybrid_mode_uses_search_hybrid() -> None:
    hybrid_result = [make_retrieved("h#0")]
    vector_store = FakeHybridVectorStore(hybrid_results=hybrid_result)
    pipeline = RagPipeline(
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
        vector_store=vector_store,
        generator=FakeGenerator(),
        retrieval_mode="hybrid",
        sparse_encoder=FakeSparseEncoder(),
        top_k=5,
    )
    result = pipeline.query("q")
    assert result.retrieved_chunk_ids == ["h#0"]


def test_dense_mode_default_ignores_sparse_encoder_absence() -> None:
    # Default retrieval_mode="dense" never requires a sparse_encoder.
    vector_store = FakeVectorStore(search_results=[make_retrieved("d#0")])
    pipeline = RagPipeline(
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
        vector_store=vector_store,
        generator=FakeGenerator(),
    )
    result = pipeline.query("q")
    assert result.retrieved_chunk_ids == ["d#0"]


def test_index_corpus_fits_and_upserts_sparse_vectors_in_hybrid_mode(tmp_path: Path) -> None:
    (tmp_path / "doc1.txt").write_text("hello world", encoding="utf-8")
    sparse_encoder = FakeSparseEncoder()
    vector_store = FakeHybridVectorStore()
    pipeline = RagPipeline(
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
        vector_store=vector_store,
        generator=FakeGenerator(),
        retrieval_mode="hybrid",
        sparse_encoder=sparse_encoder,
    )
    pipeline.index_corpus(tmp_path)
    assert len(sparse_encoder.fit_calls) == 1
    assert len(vector_store.sparse_upserted) == 1


class FakeReranker:
    def __init__(self, reordered: list[RetrievedChunk]) -> None:
        self._reordered = reordered
        self.received_query: str | None = None
        self.received_chunks: list[RetrievedChunk] | None = None

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        self.received_query = query
        self.received_chunks = chunks
        return self._reordered[:top_k]


def test_reranker_narrows_fanout_down_to_top_k() -> None:
    fanout_results = [make_retrieved(f"c#{i}") for i in range(20)]
    vector_store = FakeVectorStore(search_results=fanout_results)
    reranker = FakeReranker(reordered=[fanout_results[17], fanout_results[3]])
    pipeline = RagPipeline(
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
        vector_store=vector_store,
        generator=FakeGenerator(),
        top_k=2,
        reranker=reranker,
        retrieval_fanout=20,
    )
    result = pipeline.query("q")
    assert result.retrieved_chunk_ids == ["c#17", "c#3"]
    assert reranker.received_chunks is not None
    assert len(reranker.received_chunks) == 20


def test_reranker_receives_original_question_not_rewritten_query() -> None:
    vector_store = FakeVectorStore(search_results=[make_retrieved("a#0")])
    reranker = FakeReranker(reordered=[make_retrieved("a#0")])
    pipeline = RagPipeline(
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
        vector_store=vector_store,
        generator=FakeGenerator(),
        reranker=reranker,
    )
    pipeline.query("original question")
    assert reranker.received_query == "original question"


class FakeQueryRewriter:
    def __init__(self, rewritten: str) -> None:
        self._rewritten = rewritten
        self.received_question: str | None = None

    def rewrite(self, question: str) -> str:
        self.received_question = question
        return self._rewritten


class RecordingEmbedder(FakeEmbedder):
    def __init__(self) -> None:
        self.embedded_queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        self.embedded_queries.append(text)
        return [0.1, 0.2, 0.3, 0.4]


def test_query_rewriter_changes_the_embedded_query_only() -> None:
    embedder = RecordingEmbedder()
    rewriter = FakeQueryRewriter(rewritten="rewritten retrieval query")
    generator = FakeGenerator()
    pipeline = RagPipeline(
        chunker=FakeChunker(),
        embedder=embedder,
        vector_store=FakeVectorStore(search_results=[make_retrieved("a#0")]),
        generator=generator,
        query_rewriter=rewriter,
    )
    pipeline.query("original user question")

    assert rewriter.received_question == "original user question"
    assert embedder.embedded_queries == ["rewritten retrieval query"]
    # The generator must still see the user's original question, never the rewrite.
    assert generator.received_chunks == [make_retrieved("a#0")]


def test_latency_ms_has_same_required_keys_with_all_phase4_features_enabled() -> None:
    vector_store = FakeHybridVectorStore(hybrid_results=[make_retrieved("a#0")])
    pipeline = RagPipeline(
        chunker=FakeChunker(),
        embedder=FakeEmbedder(),
        vector_store=vector_store,
        generator=FakeGenerator(),
        retrieval_mode="hybrid",
        sparse_encoder=FakeSparseEncoder(),
        reranker=FakeReranker(reordered=[make_retrieved("a#0")]),
        query_rewriter=FakeQueryRewriter(rewritten="rewritten"),
        top_k=1,
    )
    result = pipeline.query("q")
    assert set(result.latency_ms.keys()) == {"embed", "retrieve", "generate", "total"}
    assert all(isinstance(v, float) for v in result.latency_ms.values())
