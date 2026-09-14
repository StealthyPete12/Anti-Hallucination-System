"""Unit tests for eval/dataset.py: the typed golden-set loader eval.run consumes.

Distinct from tests/test_golden_set.py, which guards the real
data/golden_set.jsonl file's schema/integrity - these tests exercise the
loader itself against small, hand-built fixture files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.dataset import DatasetError, GoldenQuestion, load_golden_set


def write_jsonl(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def make_line(**overrides: object) -> str:
    import json

    record = {
        "id": "q001",
        "type": "single_hop",
        "question": "What is MVCC?",
        "reference_answer": "Multiversion Concurrency Control.",
        "supporting_chunk_ids": ["postgres_mvcc_intro#0"],
        "tags": ["mvcc"],
    }
    record.update(overrides)
    return json.dumps(record)


def test_load_golden_set_happy_path(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "golden.jsonl", [make_line()])
    questions = load_golden_set(path)
    assert questions == [
        GoldenQuestion(
            id="q001",
            type="single_hop",
            question="What is MVCC?",
            reference_answer="Multiversion Concurrency Control.",
            supporting_chunk_ids=["postgres_mvcc_intro#0"],
            tags=["mvcc"],
        )
    ]


def test_load_golden_set_skips_blank_lines(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "golden.jsonl", [make_line(id="q001"), "", "   ", make_line(id="q002")]
    )
    questions = load_golden_set(path)
    assert [q.id for q in questions] == ["q001", "q002"]


def test_load_golden_set_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetError):
        load_golden_set(tmp_path / "does_not_exist.jsonl")


def test_load_golden_set_malformed_json_raises(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "golden.jsonl", ["{not valid json"])
    with pytest.raises(DatasetError):
        load_golden_set(path)


def test_load_golden_set_missing_field_raises(tmp_path: Path) -> None:
    import json

    record = {"id": "q001", "type": "single_hop", "question": "q"}  # missing several fields
    path = write_jsonl(tmp_path / "golden.jsonl", [json.dumps(record)])
    with pytest.raises(DatasetError):
        load_golden_set(path)


def test_load_golden_set_empty_file_raises(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "golden.jsonl", [])
    with pytest.raises(DatasetError):
        load_golden_set(path)


def test_requires_refusal_true_for_unanswerable() -> None:
    q = GoldenQuestion(
        id="q1",
        type="unanswerable",
        question="q",
        reference_answer="r",
        supporting_chunk_ids=[],
        tags=[],
    )
    assert q.requires_refusal is True


def test_requires_refusal_true_for_false_premise() -> None:
    q = GoldenQuestion(
        id="q1",
        type="false_premise",
        question="q",
        reference_answer="r",
        supporting_chunk_ids=["a#0"],
        tags=[],
    )
    assert q.requires_refusal is True


def test_requires_refusal_false_for_single_hop() -> None:
    q = GoldenQuestion(
        id="q1",
        type="single_hop",
        question="q",
        reference_answer="r",
        supporting_chunk_ids=["a#0"],
        tags=[],
    )
    assert q.requires_refusal is False


def test_requires_refusal_false_for_multi_hop() -> None:
    q = GoldenQuestion(
        id="q1",
        type="multi_hop",
        question="q",
        reference_answer="r",
        supporting_chunk_ids=["a#0", "b#1"],
        tags=[],
    )
    assert q.requires_refusal is False


def test_load_real_golden_set_file() -> None:
    """Sanity check against the actual project golden set (already schema-tested
    by tests/test_golden_set.py) - this loader must be able to read it too."""
    repo_root = Path(__file__).resolve().parent.parent.parent
    questions = load_golden_set(repo_root / "data" / "golden_set.jsonl")
    assert len(questions) == 50
    assert any(q.requires_refusal for q in questions)
    assert any(not q.requires_refusal for q in questions)
