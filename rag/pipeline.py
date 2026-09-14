"""End-to-end RAG pipeline orchestration.

Wires together a Chunker, Embedder, VectorStore, and Generator (each typed
against the Protocol contracts in ``rag.protocols``) into the two operations
the rest of the system needs: indexing a corpus, and answering a query.
Every component is injected through the constructor, so any implementation
satisfying the relevant Protocol can be swapped in without touching this
file - including test fakes.

Phase 4 additions - all optional, all off by default, so the zero-arg
construction path is byte-for-byte the Phase 1 baseline:

- ``retrieval_mode`` ("dense" | "sparse" | "hybrid") plus ``sparse_encoder``
  route retrieval through :meth:`rag.stores.qdrant.QdrantVectorStore.search_sparse`
  / ``search_hybrid`` instead of dense-only ``search``.
- ``reranker`` + ``retrieval_fanout`` implement the
  Query -> top-N retrieval -> cross-encoder -> top-K -> Generator pipeline.
- ``query_rewriter`` rewrites the *retrieval* query only; the generator
  always sees the user's original question.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from rag.citation_validator import validate_citations
from rag.metrics import metrics_recorder
from rag.models import PipelineResult, RetrievedChunk
from rag.protocols import (
    Chunker,
    Embedder,
    Generator,
    QueryRewriter,
    Reranker,
    SparseEncoder,
    VectorStore,
)
from rag.telemetry import get_tracer

logger = logging.getLogger("rag.pipeline")

RETRIEVAL_MODES = ("dense", "sparse", "hybrid")


class PipelineConfigError(ValueError):
    """Raised when the pipeline is constructed with an invalid Phase 4 configuration."""


class RagPipeline:
    """Orchestrates chunking -> embedding -> vector storage -> retrieval -> generation."""

    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
        generator: Generator,
        top_k: int = 5,
        *,
        retrieval_mode: str = "dense",
        sparse_encoder: SparseEncoder | None = None,
        reranker: Reranker | None = None,
        retrieval_fanout: int | None = None,
        rrf_k: int = 60,
        query_rewriter: QueryRewriter | None = None,
    ) -> None:
        if retrieval_mode not in RETRIEVAL_MODES:
            raise PipelineConfigError(
                f"retrieval_mode must be one of {RETRIEVAL_MODES}, got {retrieval_mode!r}"
            )
        if retrieval_mode in ("sparse", "hybrid") and sparse_encoder is None:
            raise PipelineConfigError(
                f"retrieval_mode={retrieval_mode!r} requires a sparse_encoder"
            )

        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator
        self.top_k = top_k
        self.retrieval_mode = retrieval_mode
        self.sparse_encoder = sparse_encoder
        self.reranker = reranker
        self.retrieval_fanout = retrieval_fanout or max(top_k, 20)
        self.rrf_k = rrf_k
        self.query_rewriter = query_rewriter

    def index_corpus(self, corpus_dir: Path) -> int:
        """Chunk every ``*.txt`` document in ``corpus_dir``, embed, and upsert.

        When ``retrieval_mode`` is "sparse"/"hybrid", also fits the sparse
        encoder on the full corpus and upserts a BM25 sparse vector for every
        chunk alongside its dense vector. Returns the total number of chunks
        indexed.
        """
        doc_paths = sorted(corpus_dir.glob("*.txt"))
        if not doc_paths:
            logger.warning("No .txt documents found in %s", corpus_dir)
            return 0

        if not self.vector_store.collection_exists():
            self.vector_store.create_collection(vector_size=self.embedder.dimension)

        all_chunks = []
        total = 0
        for doc_path in doc_paths:
            text = doc_path.read_text(encoding="utf-8")
            chunks = self.chunker.chunk_document(
                doc_id=doc_path.stem, text=text, source_path=str(doc_path)
            )
            if not chunks:
                continue
            vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
            self.vector_store.upsert(chunks, vectors)
            all_chunks.extend(chunks)
            total += len(chunks)

        if self.retrieval_mode in ("sparse", "hybrid") and all_chunks:
            assert self.sparse_encoder is not None  # enforced in __init__
            fit = getattr(self.sparse_encoder, "fit", None)
            if callable(fit):
                fit([chunk.text for chunk in all_chunks])
            sparse_vectors = [
                self.sparse_encoder.encode_document(chunk.text) for chunk in all_chunks
            ]
            self.vector_store.upsert_sparse(all_chunks, sparse_vectors)  # type: ignore[attr-defined]

        logger.info("Indexed %d chunk(s) from %d document(s)", total, len(doc_paths))
        return total

    def retrieve(
        self, question: str, trace_id: str | None = None
    ) -> tuple[list[RetrievedChunk], dict[str, float]]:
        """Embed + retrieve (+ rerank) for ``question``, without generation.

        Split out from :meth:`query` so retrieval-only evaluation (Phase 4
        experiments measuring Recall@5/MRR/nDCG@10 without an LLM judge or
        generator) can call exactly the same code path the production
        endpoint uses. Returns ``(retrieved_chunks, stage_latencies_ms)``.

        Phase 5: wraps the embed+search step in a "retrieve" span and the
        (possibly no-op) reranking step in its own "rerank" span, so both
        are visible in Phoenix independent of the ``latency_ms`` dict
        returned to callers (unchanged since Phase 1/4).
        """
        tracer = get_tracer()
        stages: dict[str, float] = {}

        t0 = time.perf_counter()
        retrieval_query = question
        if self.query_rewriter is not None:
            retrieval_query = self.query_rewriter.rewrite(question)
        stages["rewrite"] = round((time.perf_counter() - t0) * 1000, 1)

        fetch_k = self.retrieval_fanout if self.reranker is not None else self.top_k

        with tracer.start_as_current_span("retrieve") as span:
            span.set_attribute("trace_id", trace_id or "")
            span.set_attribute("retrieval.mode", self.retrieval_mode)
            span.set_attribute("retrieval.top_k", self.top_k)
            span.set_attribute("retrieval.fanout", fetch_k)

            t0 = time.perf_counter()
            query_vector = self.embedder.embed_query(retrieval_query)
            stages["embed"] = round((time.perf_counter() - t0) * 1000, 1)

            t0 = time.perf_counter()
            if self.retrieval_mode == "dense":
                retrieved = self.vector_store.search(query_vector, top_k=fetch_k)
            elif self.retrieval_mode == "sparse":
                assert self.sparse_encoder is not None
                sparse_query = self.sparse_encoder.encode_query(retrieval_query)
                retrieved = self.vector_store.search_sparse(sparse_query, top_k=fetch_k)  # type: ignore[attr-defined]
            else:  # hybrid
                assert self.sparse_encoder is not None
                sparse_query = self.sparse_encoder.encode_query(retrieval_query)
                retrieved = self.vector_store.search_hybrid(  # type: ignore[attr-defined]
                    query_vector, sparse_query, top_k=fetch_k, fanout=fetch_k, rrf_k=self.rrf_k
                )
            stages["retrieve"] = round((time.perf_counter() - t0) * 1000, 1)
            span.set_attribute("retrieval.embed_ms", stages["embed"])
            span.set_attribute("retrieval.search_ms", stages["retrieve"])
            span.set_attribute("retrieval.num_results", len(retrieved))

        with tracer.start_as_current_span("rerank") as span:
            span.set_attribute("trace_id", trace_id or "")
            span.set_attribute("reranker.enabled", self.reranker is not None)
            span.set_attribute("rerank.input_count", len(retrieved))
            t0 = time.perf_counter()
            if self.reranker is not None:
                span.set_attribute("reranker.model", getattr(self.reranker, "model_name", ""))
                retrieved = self.reranker.rerank(question, retrieved, top_k=self.top_k)
            stages["rerank"] = round((time.perf_counter() - t0) * 1000, 1)
            span.set_attribute("rerank.output_count", len(retrieved))

        return retrieved, stages

    def query(self, question: str, trace_id: str | None = None) -> PipelineResult:
        """Answer ``question`` end to end, returning per-stage timings and citations."""
        trace_id = trace_id or str(uuid.uuid4())
        tracer = get_tracer()

        with tracer.start_as_current_span("rag.query") as root_span:
            root_span.set_attribute("trace_id", trace_id)
            root_span.set_attribute("question", question)
            root_span.set_attribute("retrieval.mode", self.retrieval_mode)
            root_span.set_attribute("reranker.enabled", self.reranker is not None)
            root_span.set_attribute("query_rewrite.enabled", self.query_rewriter is not None)

            t_start = time.perf_counter()
            logger.info(
                "trace_id=%s Received query: %r mode=%s rerank=%s rewrite=%s",
                trace_id,
                question,
                self.retrieval_mode,
                self.reranker is not None,
                self.query_rewriter is not None,
            )

            retrieved, stages = self.retrieve(question, trace_id=trace_id)
            retrieved_chunk_ids = [chunk.chunk_id for chunk in retrieved]
            logger.info(
                "trace_id=%s Retrieved %d chunk(s) "
                "(embed=%.1fms retrieve=%.1fms rerank=%.1fms): %s",
                trace_id,
                len(retrieved),
                stages["embed"],
                stages["retrieve"],
                stages["rerank"],
                retrieved_chunk_ids,
            )

            with tracer.start_as_current_span("generate") as gen_span:
                gen_span.set_attribute("trace_id", trace_id)
                gen_span.set_attribute("llm.model", getattr(self.generator, "model", ""))
                gen_span.set_attribute("generate.num_chunks", len(retrieved))
                t0 = time.perf_counter()
                try:
                    result = self.generator.generate(question, retrieved)
                except Exception:
                    logger.exception("trace_id=%s Generation failed", trace_id)
                    raise
                generate_ms = (time.perf_counter() - t0) * 1000
                gen_span.set_attribute("request.input_tokens", result.usage.prompt_tokens)
                gen_span.set_attribute("request.output_tokens", result.usage.completion_tokens)
                gen_span.set_attribute("request.cost_usd", result.cost_usd)

            logger.info(
                "trace_id=%s Generated answer with citations=%s in %.1fms",
                trace_id,
                result.citations,
                generate_ms,
            )

            citation_check = validate_citations(result.answer, retrieved_chunk_ids)
            if not citation_check.citation_validation_passed:
                logger.warning(
                    "trace_id=%s Citation validation failed: invalid_citations=%s",
                    trace_id,
                    citation_check.invalid_citations,
                )
            root_span.set_attribute(
                "citation_validation_passed", citation_check.citation_validation_passed
            )
            root_span.set_attribute("invalid_citations", citation_check.invalid_citations)

            total_ms = (time.perf_counter() - t_start) * 1000
            latency_ms = {
                "embed": stages["embed"],
                "retrieve": stages["retrieve"] + stages["rerank"] + stages["rewrite"],
                "generate": round(generate_ms, 1),
                "total": round(total_ms, 1),
            }
            root_span.set_attribute("request.cost_usd", result.cost_usd)
            root_span.set_attribute("request.input_tokens", result.usage.prompt_tokens)
            root_span.set_attribute("request.output_tokens", result.usage.completion_tokens)
            root_span.set_attribute("latency.total_ms", latency_ms["total"])
            logger.info("trace_id=%s Completed in %.1fms", trace_id, total_ms)

        metrics_recorder.record(cost_usd=result.cost_usd, latency_ms=latency_ms["total"])

        return PipelineResult(
            answer=result.answer,
            citations=result.citations,
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieved_chunks=retrieved,
            latency_ms=latency_ms,
            usage=result.usage,
            cost_usd=result.cost_usd,
            trace_id=trace_id,
            citation_validation_passed=citation_check.citation_validation_passed,
            invalid_citations=citation_check.invalid_citations,
        )
