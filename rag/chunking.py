"""Recursive character text splitter for chunking source documents.

Splits on a cascade of separators (paragraph -> line -> sentence -> word ->
character), falling back to a finer separator only where a piece is still
too large for the configured chunk size. Chunk boundaries are a pure
function of the input text and configuration, so re-chunking an unchanged
source file always reproduces byte-identical chunk_ids and text.

Chunk size and overlap are measured in *tokens* via an injectable
``token_counter`` callable rather than characters or words. This module
defaults to a fast, dependency-free whitespace approximation; production
ingestion should inject the embedding model's own tokenizer (see
``rag.embedders.local.LocalBGEEmbedder.count_tokens``) so chunk boundaries
respect that model's real max-sequence-length.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

from rag.models import Chunk

logger = logging.getLogger("rag.chunking")

TokenCounter = Callable[[str], int]

DEFAULT_CHUNK_SIZE_TOKENS = 512
DEFAULT_OVERLAP_RATIO = 0.15
DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")

# Matches short, numbered/lettered section headings like "13.1. INTRODUCTION #"
# or "5.10.1. CREATING A SCHEMA #", which is how this corpus's source pages
# render their headings as plain text.
_HEADING_RE = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+.{2,100}?\s*#?\s*$")


def approximate_token_counter(text: str) -> int:
    """Whitespace-based token count approximation.

    Fast and dependency-free; used as the chunker's default and in tests.
    """
    return len(text.split())


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 100:
        return False
    if not _HEADING_RE.match(stripped):
        return False
    letters = [c for c in stripped if c.isalpha()]
    if not letters:
        return False
    return sum(c.isupper() for c in letters) / len(letters) > 0.6


def _heading_before(text: str, position: int, fallback: str) -> str:
    """Return the nearest heading line at or before ``position`` in ``text``."""
    preceding = text[:position]
    for line in reversed(preceding.splitlines()):
        if _looks_like_heading(line):
            return line.strip().rstrip("#").strip()
    return fallback


def _split_with_separator(text: str, separator: str) -> list[str]:
    if separator == "":
        return list(text)
    return text.split(separator)


def _merge_splits(
    splits: list[str],
    separator: str,
    chunk_size: int,
    overlap: int,
    count_tokens: TokenCounter,
) -> list[str]:
    """Greedily pack small pieces into chunks up to ``chunk_size``, keeping
    the trailing ``overlap`` tokens of each chunk at the start of the next."""
    merged: list[str] = []
    current: list[str] = []
    current_len = 0
    sep_len = count_tokens(separator) if separator else 0

    for piece in splits:
        piece_len = count_tokens(piece)
        projected = current_len + piece_len + (sep_len if current else 0)
        if projected > chunk_size and current:
            joined = separator.join(current)
            if joined.strip():
                merged.append(joined)
            while current and (
                current_len > overlap
                or current_len + piece_len + (sep_len if len(current) > 1 else 0) > chunk_size
            ):
                dropped = current.pop(0)
                current_len -= count_tokens(dropped) + (sep_len if current else 0)
        current.append(piece)
        current_len += piece_len + (sep_len if len(current) > 1 else 0)

    joined = separator.join(current)
    if joined.strip():
        merged.append(joined)
    return merged


def _recursive_split(
    text: str,
    separators: tuple[str, ...],
    chunk_size: int,
    overlap: int,
    count_tokens: TokenCounter,
) -> list[str]:
    separator = separators[-1]
    remaining_separators: tuple[str, ...] = ()
    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            separator = sep
            remaining_separators = separators[i + 1 :]
            break

    splits = _split_with_separator(text, separator)

    good_splits: list[str] = []
    chunks: list[str] = []
    for piece in splits:
        if count_tokens(piece) <= chunk_size:
            good_splits.append(piece)
            continue
        if good_splits:
            chunks.extend(_merge_splits(good_splits, separator, chunk_size, overlap, count_tokens))
            good_splits = []
        if remaining_separators:
            chunks.extend(
                _recursive_split(piece, remaining_separators, chunk_size, overlap, count_tokens)
            )
        else:
            logger.warning(
                "Piece of %d tokens has no finer separator to split on; keeping as one chunk",
                count_tokens(piece),
            )
            chunks.append(piece)

    if good_splits:
        chunks.extend(_merge_splits(good_splits, separator, chunk_size, overlap, count_tokens))

    return chunks


@dataclass(frozen=True)
class ChunkingConfig:
    """Validated configuration for a single chunking run."""

    chunk_size_tokens: int = DEFAULT_CHUNK_SIZE_TOKENS
    overlap_ratio: float = DEFAULT_OVERLAP_RATIO
    separators: tuple[str, ...] = DEFAULT_SEPARATORS

    def __post_init__(self) -> None:
        if self.chunk_size_tokens <= 0:
            raise ValueError(f"chunk_size_tokens must be positive, got {self.chunk_size_tokens}")
        if not 0 <= self.overlap_ratio < 1:
            raise ValueError(f"overlap_ratio must be in [0, 1), got {self.overlap_ratio}")

    @property
    def overlap_tokens(self) -> int:
        return int(self.chunk_size_tokens * self.overlap_ratio)


class RecursiveCharacterChunker:
    """Recursive character text splitter producing token-bounded chunks with metadata."""

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        token_counter: TokenCounter = approximate_token_counter,
    ) -> None:
        self.config = config or ChunkingConfig()
        self.count_tokens = token_counter

    def chunk_document(self, *, doc_id: str, text: str, source_path: str) -> list[Chunk]:
        text = text.strip()
        if not text:
            logger.warning("Document %s is empty; producing no chunks", doc_id)
            return []

        pieces = _recursive_split(
            text,
            self.config.separators,
            self.config.chunk_size_tokens,
            self.config.overlap_tokens,
            self.count_tokens,
        )

        chunks: list[Chunk] = []
        current_heading = ""
        for piece in pieces:
            piece_stripped = piece.strip()
            if not piece_stripped:
                continue
            located = text.find(piece_stripped[:80])
            if located != -1:
                current_heading = _heading_before(text, located, current_heading)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc_id}#{len(chunks)}",
                    doc_id=doc_id,
                    chunk_index=len(chunks),
                    text=piece_stripped,
                    source_path=source_path,
                    heading=current_heading,
                )
            )

        if not chunks:
            logger.warning("Document %s produced no chunks", doc_id)
        else:
            logger.info("Chunked %s into %d chunk(s)", doc_id, len(chunks))
        return chunks
