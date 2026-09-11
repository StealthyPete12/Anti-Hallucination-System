"""Deterministic ingestion pipeline: corpus text files -> chunked JSONL.

Reads every ``*.txt`` document in a corpus directory, splits each one into
overlapping word-based chunks, and writes one JSON record per chunk to an
output JSONL file. Chunk IDs are derived from the document ID (the file
stem) and the chunk's position within that document, so re-running the
pipeline against unchanged source files always produces the same IDs.

Usage:
    python -m rag.ingest --corpus-dir data/corpus --output data/index/chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger("rag.ingest")

DEFAULT_CHUNK_SIZE = 220
DEFAULT_OVERLAP = 40


@dataclass(frozen=True)
class Chunk:
    """A single chunk of a source document, ready to be written to JSONL."""

    chunk_id: str
    text: str
    doc_id: str
    chunk_index: int
    num_words: int


class IngestConfig:
    """Validated configuration for a single ingestion run."""

    def __init__(
        self, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_OVERLAP
    ) -> None:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {overlap}")
        if overlap >= chunk_size:
            raise ValueError(
                f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size}) "
                "or chunking will never make forward progress"
            )
        self.chunk_size = chunk_size
        self.overlap = overlap


def discover_documents(corpus_dir: Path) -> list[Path]:
    """Return every ``*.txt`` file in ``corpus_dir``, sorted for determinism."""
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Corpus directory does not exist: {corpus_dir}")
    return sorted(corpus_dir.glob("*.txt"))


def chunk_text(text: str, config: IngestConfig) -> Iterator[str]:
    """Split ``text`` into overlapping, whitespace-tokenized chunks.

    Chunking is purely a function of the token stream and config, so the
    same input always yields the same chunk boundaries.
    """
    words = text.split()
    if not words:
        return

    step = config.chunk_size - config.overlap
    start = 0
    n = len(words)
    while start < n:
        end = min(start + config.chunk_size, n)
        yield " ".join(words[start:end])
        if end == n:
            break
        start += step


def chunk_document(doc_path: Path, config: IngestConfig) -> list[Chunk]:
    """Read one document and return its chunks with stable, ordered IDs."""
    doc_id = doc_path.stem
    try:
        text = doc_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.error("Failed to read %s: %s", doc_path, exc)
        return []

    chunks = [
        Chunk(
            chunk_id=f"{doc_id}#{i}",
            text=chunk_str,
            doc_id=doc_id,
            chunk_index=i,
            num_words=len(chunk_str.split()),
        )
        for i, chunk_str in enumerate(chunk_text(text, config))
    ]

    if not chunks:
        logger.warning("Document %s produced no chunks (empty file?)", doc_path)
    else:
        logger.info("Chunked %s into %d chunk(s)", doc_id, len(chunks))
    return chunks


def ingest_corpus(corpus_dir: Path, config: IngestConfig) -> list[Chunk]:
    """Chunk every document in ``corpus_dir`` and return all chunks in order."""
    doc_paths = discover_documents(corpus_dir)
    if not doc_paths:
        logger.warning("No .txt documents found in %s", corpus_dir)

    all_chunks: list[Chunk] = []
    for doc_path in doc_paths:
        all_chunks.extend(chunk_document(doc_path, config))
    return all_chunks


def write_chunks(chunks: list[Chunk], output_path: Path) -> None:
    """Write chunks to ``output_path`` as newline-delimited JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    logger.info("Wrote %d chunks to %s", len(chunks), output_path)


def run(corpus_dir: Path, output_path: Path, config: IngestConfig) -> list[Chunk]:
    """Run the full ingestion pipeline end to end."""
    logger.info(
        "Starting ingestion: corpus_dir=%s chunk_size=%d overlap=%d",
        corpus_dir,
        config.chunk_size,
        config.overlap,
    )
    chunks = ingest_corpus(corpus_dir, config)
    write_chunks(chunks, output_path)
    return chunks


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk a text corpus into JSONL for RAG evaluation."
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path("data/corpus"),
        help="Directory containing *.txt source documents (default: data/corpus)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/index/chunks.jsonl"),
        help="Output JSONL path (default: data/index/chunks.jsonl)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Chunk size in whitespace-delimited words (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help=f"Word overlap between consecutive chunks (default: {DEFAULT_OVERLAP})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    config = IngestConfig(chunk_size=args.chunk_size, overlap=args.overlap)
    run(args.corpus_dir, args.output, config)


if __name__ == "__main__":
    main()
