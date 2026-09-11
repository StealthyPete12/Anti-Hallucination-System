"""Validation tests for the golden benchmark set.

These tests are the guardrail against evaluation drift: they enforce the
golden set's schema, its required type distribution, that every supporting
chunk reference actually resolves against the current chunk index, and
basic integrity properties (no duplicates, no empty fields, no malformed
records). If any of these fail, the golden set can no longer be trusted
as a benchmark and must be fixed before use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
GOLDEN_SET_PATH = REPO_ROOT / "data" / "golden_set.jsonl"
CHUNKS_PATH = REPO_ROOT / "data" / "index" / "chunks.jsonl"

REQUIRED_FIELDS = {"id", "type", "question", "reference_answer", "supporting_chunk_ids", "tags"}
VALID_TYPES = {"single_hop", "multi_hop", "unanswerable", "false_premise"}
EXPECTED_DISTRIBUTION = {
    "single_hop": 25,
    "multi_hop": 10,
    "unanswerable": 10,
    "false_premise": 5,
}
EXPECTED_TOTAL = 50


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                pytest.fail(f"{path.name}:{line_num}: malformed JSON: {exc}")
    return records


@pytest.fixture(scope="module")
def golden_set() -> list[dict[str, Any]]:
    assert GOLDEN_SET_PATH.exists(), f"Golden set not found at {GOLDEN_SET_PATH}"
    return _load_jsonl(GOLDEN_SET_PATH)


@pytest.fixture(scope="module")
def chunk_ids() -> set[str]:
    assert CHUNKS_PATH.exists(), (
        f"Chunk index not found at {CHUNKS_PATH}. Run `python -m rag.ingest` first."
    )
    chunks = _load_jsonl(CHUNKS_PATH)
    for chunk in chunks:
        assert "chunk_id" in chunk, "Malformed chunk record missing chunk_id"
        assert "text" in chunk, f"Chunk {chunk.get('chunk_id')} missing text"
        assert "doc_id" in chunk, f"Chunk {chunk.get('chunk_id')} missing doc_id"
    return {c["chunk_id"] for c in chunks}


# ---------------------------------------------------------------------------
# A. Schema validation
# ---------------------------------------------------------------------------


def test_golden_set_file_exists() -> None:
    assert GOLDEN_SET_PATH.exists()


def test_every_entry_has_required_fields(golden_set: list[dict[str, Any]]) -> None:
    for entry in golden_set:
        missing = REQUIRED_FIELDS - entry.keys()
        assert not missing, f"Entry {entry.get('id', '?')} missing fields: {missing}"


def test_field_types_are_correct(golden_set: list[dict[str, Any]]) -> None:
    for entry in golden_set:
        eid = entry.get("id", "?")
        assert isinstance(entry["id"], str), f"{eid}: id must be a string"
        assert isinstance(entry["type"], str), f"{eid}: type must be a string"
        assert isinstance(entry["question"], str), f"{eid}: question must be a string"
        assert isinstance(entry["reference_answer"], str), (
            f"{eid}: reference_answer must be a string"
        )
        assert isinstance(entry["supporting_chunk_ids"], list), (
            f"{eid}: supporting_chunk_ids must be a list"
        )
        assert isinstance(entry["tags"], list), f"{eid}: tags must be a list"
        assert all(isinstance(c, str) for c in entry["supporting_chunk_ids"]), (
            f"{eid}: supporting_chunk_ids must all be strings"
        )
        assert all(isinstance(t, str) for t in entry["tags"]), f"{eid}: tags must all be strings"


# ---------------------------------------------------------------------------
# B. Distribution validation
# ---------------------------------------------------------------------------


def test_total_record_count(golden_set: list[dict[str, Any]]) -> None:
    assert len(golden_set) == EXPECTED_TOTAL, (
        f"Expected {EXPECTED_TOTAL} golden set records, found {len(golden_set)}"
    )


def test_question_type_distribution(golden_set: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for entry in golden_set:
        counts[entry["type"]] = counts.get(entry["type"], 0) + 1
    assert counts == EXPECTED_DISTRIBUTION, (
        f"Distribution mismatch: {counts} != {EXPECTED_DISTRIBUTION}"
    )


def test_all_types_are_valid(golden_set: list[dict[str, Any]]) -> None:
    for entry in golden_set:
        assert entry["type"] in VALID_TYPES, f"{entry['id']}: invalid type {entry['type']!r}"


# ---------------------------------------------------------------------------
# C. Chunk reference validation
# ---------------------------------------------------------------------------


def test_supporting_chunks_exist(golden_set: list[dict[str, Any]], chunk_ids: set[str]) -> None:
    for entry in golden_set:
        for chunk_id in entry["supporting_chunk_ids"]:
            assert chunk_id in chunk_ids, (
                f"{entry['id']}: supporting_chunk_id {chunk_id!r} does not exist in chunks.jsonl"
            )


def test_answerable_entries_have_supporting_chunks(golden_set: list[dict[str, Any]]) -> None:
    for entry in golden_set:
        if entry["type"] in ("single_hop", "multi_hop", "false_premise"):
            assert entry["supporting_chunk_ids"], (
                f"{entry['id']} ({entry['type']}) must have at least one supporting chunk"
            )


def test_multi_hop_entries_cite_multiple_chunks(golden_set: list[dict[str, Any]]) -> None:
    for entry in golden_set:
        if entry["type"] == "multi_hop":
            assert len(entry["supporting_chunk_ids"]) >= 2, (
                f"{entry['id']}: multi_hop questions must cite at least 2 supporting chunks, "
                f"got {len(entry['supporting_chunk_ids'])}"
            )


def test_unanswerable_entries_have_no_supporting_chunks(golden_set: list[dict[str, Any]]) -> None:
    for entry in golden_set:
        if entry["type"] == "unanswerable":
            assert entry["supporting_chunk_ids"] == [], (
                f"{entry['id']}: unanswerable questions must have empty supporting_chunk_ids, "
                f"got {entry['supporting_chunk_ids']}"
            )


# ---------------------------------------------------------------------------
# D. Integrity validation
# ---------------------------------------------------------------------------


def test_no_duplicate_ids(golden_set: list[dict[str, Any]]) -> None:
    ids = [entry["id"] for entry in golden_set]
    duplicates = {i for i in ids if ids.count(i) > 1}
    assert not duplicates, f"Duplicate ids found: {duplicates}"


def test_no_duplicate_questions(golden_set: list[dict[str, Any]]) -> None:
    questions = [entry["question"].strip().lower() for entry in golden_set]
    duplicates = {q for q in questions if questions.count(q) > 1}
    assert not duplicates, f"Duplicate questions found: {duplicates}"


def test_no_empty_questions_or_answers(golden_set: list[dict[str, Any]]) -> None:
    for entry in golden_set:
        assert entry["question"].strip(), f"{entry['id']}: empty question"
        assert entry["reference_answer"].strip(), f"{entry['id']}: empty reference_answer"


def test_no_empty_tags(golden_set: list[dict[str, Any]]) -> None:
    for entry in golden_set:
        assert entry["tags"], f"{entry['id']}: must have at least one tag"
        assert all(t.strip() for t in entry["tags"]), (
            f"{entry['id']}: tags must not be blank strings"
        )


def test_unanswerable_entries_use_refusal_language(golden_set: list[dict[str, Any]]) -> None:
    for entry in golden_set:
        if entry["type"] == "unanswerable":
            answer = entry["reference_answer"].lower()
            assert (
                "does not contain enough information" in answer
                or "cannot" in answer
                or "not contain" in answer
            ), (
                f"{entry['id']}: unanswerable reference_answer should be a refusal, "
                f"got: {entry['reference_answer']!r}"
            )


def test_ids_follow_expected_format(golden_set: list[dict[str, Any]]) -> None:
    for entry in golden_set:
        assert entry["id"].startswith("q") and entry["id"][1:].isdigit(), (
            f"Unexpected id format: {entry['id']!r}"
        )


# ---------------------------------------------------------------------------
# Chunk index sanity (protects the ingestion pipeline's output too)
# ---------------------------------------------------------------------------


def test_chunks_file_is_nonempty(chunk_ids: set[str]) -> None:
    assert len(chunk_ids) > 0, "chunks.jsonl produced no chunks"


def test_chunk_ids_are_unique(chunk_ids: set[str]) -> None:
    raw = _load_jsonl(CHUNKS_PATH)
    raw_ids = [c["chunk_id"] for c in raw]
    assert len(raw_ids) == len(chunk_ids), "Duplicate chunk_id values found in chunks.jsonl"


def test_chunk_id_format_matches_doc_id_hash_index(chunk_ids: set[str]) -> None:
    raw = _load_jsonl(CHUNKS_PATH)
    for chunk in raw:
        expected_prefix = f"{chunk['doc_id']}#"
        assert chunk["chunk_id"].startswith(expected_prefix), (
            f"chunk_id {chunk['chunk_id']!r} does not match doc_id {chunk['doc_id']!r}"
        )
        suffix = chunk["chunk_id"][len(expected_prefix) :]
        assert suffix.isdigit(), f"chunk_id {chunk['chunk_id']!r} suffix is not a numeric index"
