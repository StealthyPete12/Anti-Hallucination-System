"""Unit tests for the recursive character chunker (rag/chunking.py)."""

from __future__ import annotations

import pytest

from rag.chunking import (
    ChunkingConfig,
    RecursiveCharacterChunker,
    approximate_token_counter,
)


def make_paragraphs(n: int, words_per_para: int = 40) -> str:
    return "\n\n".join(" ".join(f"word{p}_{i}" for i in range(words_per_para)) for p in range(n))


def test_empty_document_produces_no_chunks() -> None:
    chunker = RecursiveCharacterChunker(config=ChunkingConfig(chunk_size_tokens=50))
    assert chunker.chunk_document(doc_id="doc", text="   ", source_path="doc.txt") == []


def test_small_document_produces_single_chunk() -> None:
    chunker = RecursiveCharacterChunker(config=ChunkingConfig(chunk_size_tokens=100))
    text = "This is a short document with very few words."
    chunks = chunker.chunk_document(doc_id="doc", text=text, source_path="doc.txt")
    assert len(chunks) == 1
    assert chunks[0].text == text
    assert chunks[0].chunk_id == "doc#0"


def test_chunk_ids_are_sequential_and_stable() -> None:
    chunker = RecursiveCharacterChunker(
        config=ChunkingConfig(chunk_size_tokens=20, overlap_ratio=0.1)
    )
    text = make_paragraphs(10, words_per_para=15)
    chunks = chunker.chunk_document(doc_id="mydoc", text=text, source_path="mydoc.txt")
    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_id == f"mydoc#{i}"
        assert chunk.chunk_index == i
        assert chunk.doc_id == "mydoc"
        assert chunk.source_path == "mydoc.txt"


def test_chunking_is_deterministic() -> None:
    chunker = RecursiveCharacterChunker(
        config=ChunkingConfig(chunk_size_tokens=30, overlap_ratio=0.15)
    )
    text = make_paragraphs(8, words_per_para=25)
    chunks_a = chunker.chunk_document(doc_id="d", text=text, source_path="d.txt")
    chunks_b = chunker.chunk_document(doc_id="d", text=text, source_path="d.txt")
    assert chunks_a == chunks_b


def test_chunks_respect_configured_size_budget() -> None:
    chunk_size = 40
    chunker = RecursiveCharacterChunker(
        config=ChunkingConfig(chunk_size_tokens=chunk_size, overlap_ratio=0.15)
    )
    text = make_paragraphs(20, words_per_para=15)
    chunks = chunker.chunk_document(doc_id="d", text=text, source_path="d.txt")
    # Recursive splitting can't always hit the budget exactly (e.g. a single
    # sentence longer than chunk_size), but with short paragraphs here every
    # chunk should stay close to the configured size.
    for chunk in chunks:
        assert approximate_token_counter(chunk.text) <= chunk_size + 5


def test_consecutive_chunks_overlap() -> None:
    # Short paragraphs (4 tokens each) so several fit into one chunk_size=20
    # window, leaving room for the merge logic to carry a trailing paragraph
    # forward as overlap into the next chunk.
    chunker = RecursiveCharacterChunker(
        config=ChunkingConfig(chunk_size_tokens=20, overlap_ratio=0.3)
    )
    text = make_paragraphs(10, words_per_para=4)
    chunks = chunker.chunk_document(doc_id="d", text=text, source_path="d.txt")
    assert len(chunks) > 1
    first_words = set(chunks[0].text.split())
    second_words = set(chunks[1].text.split())
    assert first_words & second_words, "expected some word overlap between consecutive chunks"


def test_default_config_matches_spec() -> None:
    config = ChunkingConfig()
    assert config.chunk_size_tokens == 512
    assert config.overlap_ratio == 0.15
    assert config.overlap_tokens == int(512 * 0.15)


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size_tokens=0)
    with pytest.raises(ValueError):
        ChunkingConfig(overlap_ratio=1.0)
    with pytest.raises(ValueError):
        ChunkingConfig(overlap_ratio=-0.1)


def test_heading_metadata_is_attached() -> None:
    chunker = RecursiveCharacterChunker(
        config=ChunkingConfig(chunk_size_tokens=30, overlap_ratio=0.1)
    )
    text = (
        "13.1. INTRODUCTION #\n\n"
        + " ".join(f"para1_word{i}" for i in range(40))
        + "\n\n13.2. DETAILS #\n\n"
        + " ".join(f"para2_word{i}" for i in range(40))
    )
    chunks = chunker.chunk_document(doc_id="d", text=text, source_path="d.txt")
    headings = {chunk.heading for chunk in chunks}
    assert "13.1. INTRODUCTION" in headings
    assert "13.2. DETAILS" in headings
    first_chunk_with_para2 = next(c for c in chunks if "para2_word0" in c.text)
    assert first_chunk_with_para2.heading == "13.2. DETAILS"


def test_custom_token_counter_is_used() -> None:
    calls: list[str] = []

    def counting_counter(text: str) -> int:
        calls.append(text)
        return approximate_token_counter(text)

    chunker = RecursiveCharacterChunker(
        config=ChunkingConfig(chunk_size_tokens=20, overlap_ratio=0.1),
        token_counter=counting_counter,
    )
    chunker.chunk_document(doc_id="d", text=make_paragraphs(3), source_path="d.txt")
    assert calls, "custom token_counter should have been invoked"
