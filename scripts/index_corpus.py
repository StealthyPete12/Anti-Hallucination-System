"""One-shot indexing: chunk data/corpus/*.txt, embed, and upsert into Qdrant.

Usage:
    uv run python -m scripts.index_corpus
    uv run python -m scripts.index_corpus --corpus-dir data/corpus --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from app.dependencies import get_pipeline
from config import settings

logger = logging.getLogger("scripts.index_corpus")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk, embed, and index the corpus into Qdrant.")
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=Path(settings.corpus_dir),
        help=f"Directory containing *.txt source documents (default: {settings.corpus_dir})",
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
    pipeline = get_pipeline()
    logger.info(
        "Indexing corpus from %s into collection %s",
        args.corpus_dir,
        settings.qdrant_collection_name,
    )
    total = pipeline.index_corpus(args.corpus_dir)
    logger.info("Done: indexed %d chunk(s) from %s", total, args.corpus_dir)


if __name__ == "__main__":
    main()
