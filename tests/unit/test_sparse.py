"""Unit tests for BM25 sparse encoding (rag/sparse.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.sparse import BM25Vectorizer, SparseEncodingError, tokenize


def test_tokenize_lowercases_and_strips_punctuation() -> None:
    assert tokenize("MVCC stands for Multi-Version Concurrency Control.") == [
        "mvcc",
        "stands",
        "for",
        "multi",
        "version",
        "concurrency",
        "control",
    ]


def test_encode_before_fit_raises() -> None:
    vectorizer = BM25Vectorizer()
    with pytest.raises(SparseEncodingError):
        vectorizer.encode_document("hello world")


def test_fit_on_empty_corpus_raises() -> None:
    with pytest.raises(ValueError):
        BM25Vectorizer().fit([])


def test_fit_builds_vocabulary_from_corpus() -> None:
    vectorizer = BM25Vectorizer().fit(["the quick fox", "the lazy dog"])
    assert set(vectorizer.vocabulary) == {"the", "quick", "fox", "lazy", "dog"}
    assert vectorizer.num_docs == 2


def test_document_vector_indices_are_within_vocabulary() -> None:
    vectorizer = BM25Vectorizer().fit(["the quick fox", "the lazy dog"])
    vector = vectorizer.encode_document("the quick fox")
    assert len(vector.indices) == len(vector.values)
    assert all(0 <= idx < len(vectorizer.vocabulary) for idx in vector.indices)


def test_rarer_term_gets_higher_weight_than_common_term() -> None:
    # "the" appears in every doc (low IDF); "fox" appears in only one (high IDF).
    corpus = ["the quick fox", "the lazy dog", "the brown cat", "the tall tree"]
    vectorizer = BM25Vectorizer().fit(corpus)
    vector = vectorizer.encode_document("the quick fox")
    weight_by_term = dict(zip(vector.indices, vector.values, strict=True))
    the_weight = weight_by_term[vectorizer.vocabulary["the"]]
    fox_weight = weight_by_term[vectorizer.vocabulary["fox"]]
    assert fox_weight > the_weight


def test_out_of_vocabulary_query_term_is_silently_dropped() -> None:
    vectorizer = BM25Vectorizer().fit(["the quick fox"])
    query_vector = vectorizer.encode_query("the quick zzznotaword")
    assert len(query_vector.indices) == 2  # "the" and "quick" only


def test_repeated_term_increases_document_weight_up_to_saturation() -> None:
    corpus = ["fox fox fox fox", "dog", "cat", "bird"]
    vectorizer = BM25Vectorizer().fit(corpus)
    single = BM25Vectorizer().fit(["fox", "dog", "cat", "bird"]).encode_document("fox")
    repeated = vectorizer.encode_document("fox fox fox fox")
    assert repeated.values[0] > single.values[0]


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    vectorizer = BM25Vectorizer().fit(["the quick fox", "the lazy dog"])
    path = tmp_path / "bm25.json"
    vectorizer.save(path)

    loaded = BM25Vectorizer.load(path)
    assert loaded.vocabulary == vectorizer.vocabulary
    assert loaded.idf == vectorizer.idf
    assert loaded.avg_doc_len == vectorizer.avg_doc_len
    assert loaded.num_docs == vectorizer.num_docs

    original_vec = vectorizer.encode_document("the quick fox")
    loaded_vec = loaded.encode_document("the quick fox")
    assert loaded_vec.indices == original_vec.indices
    assert loaded_vec.values == original_vec.values


def test_load_missing_file_raises() -> None:
    with pytest.raises(SparseEncodingError):
        BM25Vectorizer.load("/nonexistent/path/bm25.json")
