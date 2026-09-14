"""End-to-end RAG pipeline orchestration.

Wires together a Chunker, Embedder, VectorStore, and Generator (each typed
against the Protocol contracts in ``rag.protocols``) into the two operations
the rest of the system needs: indexing a corpus, and answering a query.
Every component is injected through the constructor, so any implementation
satisfying the relevant Protocol can be swapped in without touching this
file - including test fakes.
"""

from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

from rag.models import PipelineResult
from rag.protocols import Chunker, Embedder, Generator, VectorStore

logger = logging.getLogger("rag.pipeline")


class RagPipeline:
    """Orchestrates chunking -> embedding -> vector storage -> retrieval -> generation."""

    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
        generator: Generator,
        top_k: int = 5,
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator
        self.top_k = top_k

    def index_corpus(self, corpus_dir: Path) -> int:
        """Chunk every ``*.txt`` document in ``corpus_dir``, embed, and upsert.

        Returns the total number of chunks indexed.
        """
        doc_paths = sorted(corpus_dir.glob("*.txt"))
        if not doc_paths:
            logger.warning("No .txt documents found in %s", corpus_dir)
            return 0

        if not self.vector_store.collection_exists():
            self.vector_store.create_collection(vector_size=self.embedder.dimension)

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
            total += len(chunks)

        logger.info("Indexed %d chunk(s) from %d document(s)", total, len(doc_paths))
        return total

    def query(self, question: str, trace_id: str | None = None) -> PipelineResult:
        """Answer ``question`` end to end, returning per-stage timings and citations."""
        trace_id = trace_id or str(uuid.uuid4())
        t_start = time.perf_counter()
        logger.info("trace_id=%s Received query: %r", trace_id, question)

        t0 = time.perf_counter()
        query_vector = self.embedder.embed_query(question)
        embed_ms = (time.perf_counter() - t0) * 1000
        logger.info("trace_id=%s Embedded query in %.1fms", trace_id, embed_ms)

        t0 = time.perf_counter()
        retrieved = self.vector_store.search(query_vector, top_k=self.top_k)
        retrieve_ms = (time.perf_counter() - t0) * 1000
        retrieved_chunk_ids = [chunk.chunk_id for chunk in retrieved]
        logger.info(
            "trace_id=%s Retrieved %d chunk(s) in %.1fms: %s",
            trace_id,
            len(retrieved),
            retrieve_ms,
            retrieved_chunk_ids,
        )

        t0 = time.perf_counter()
        try:
            result = self.generator.generate(question, retrieved)
        except Exception:
            logger.exception("trace_id=%s Generation failed", trace_id)
            raise
        generate_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "trace_id=%s Generated answer with citations=%s in %.1fms",
            trace_id,
            result.citations,
            generate_ms,
        )

        total_ms = (time.perf_counter() - t_start) * 1000
        latency_ms = {
            "embed": round(embed_ms, 1),
            "retrieve": round(retrieve_ms, 1),
            "generate": round(generate_ms, 1),
            "total": round(total_ms, 1),
        }
        logger.info("trace_id=%s Completed in %.1fms", trace_id, total_ms)

        return PipelineResult(
            answer=result.answer,
            citations=result.citations,
            retrieved_chunk_ids=retrieved_chunk_ids,
            retrieved_chunks=retrieved,
            latency_ms=latency_ms,
            usage=result.usage,
            cost_usd=result.cost_usd,
            trace_id=trace_id,
        )
