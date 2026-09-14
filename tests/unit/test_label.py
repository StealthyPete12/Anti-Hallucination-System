"""Unit tests for the judge-validation labeling workflow (scripts/label.py).

``label_session`` takes its I/O as injectable callables (matching this
project's fakes-over-mocks convention), so the interactive workflow is
tested with a scripted list of canned responses instead of a real terminal.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.label import (
    append_labels,
    label_session,
    load_existing_labels,
    load_per_question,
    sample_questions,
)


def make_question(qid: str, verdict: str = "correct") -> dict:
    return {
        "id": qid,
        "type": "single_hop",
        "question": f"question {qid}?",
        "reference_answer": "ref",
        "generated_answer": "gen",
        "correctness": {"verdict": verdict},
        "error": None,
    }


def scripted_input(responses: list[str]):
    it = iter(responses)

    def _input(prompt: str) -> str:
        return next(it)

    return _input


def test_load_per_question_excludes_errored_questions(tmp_path: Path) -> None:
    scorecard = {
        "per_question": [
            make_question("q1"),
            {**make_question("q2"), "error": "pipeline failed"},
        ]
    }
    path = tmp_path / "scorecard.json"
    path.write_text(json.dumps(scorecard), encoding="utf-8")

    result = load_per_question(path)
    assert [q["id"] for q in result] == ["q1"]


def test_sample_questions_is_deterministic_for_a_given_seed() -> None:
    questions = [make_question(f"q{i}") for i in range(20)]
    sample_a = sample_questions(questions, sample_size=5, seed=7)
    sample_b = sample_questions(questions, sample_size=5, seed=7)
    assert [q["id"] for q in sample_a] == [q["id"] for q in sample_b]


def test_sample_questions_different_seeds_can_differ() -> None:
    questions = [make_question(f"q{i}") for i in range(20)]
    sample_a = [q["id"] for q in sample_questions(questions, sample_size=5, seed=1)]
    sample_b = [q["id"] for q in sample_questions(questions, sample_size=5, seed=2)]
    assert sample_a != sample_b


def test_sample_questions_caps_at_available_pool() -> None:
    questions = [make_question(f"q{i}") for i in range(3)]
    sample = sample_questions(questions, sample_size=40, seed=1)
    assert len(sample) == 3


def test_label_session_records_valid_label() -> None:
    items = [make_question("q1", verdict="correct")]
    new_labels = label_session(
        items, already_labeled={}, input_fn=scripted_input(["correct"]), print_fn=lambda s: None
    )
    assert len(new_labels) == 1
    assert new_labels[0]["id"] == "q1"
    assert new_labels[0]["human_label"] == "correct"
    assert new_labels[0]["judge_verdict"] == "correct"


def test_label_session_reprompts_on_invalid_label() -> None:
    items = [make_question("q1")]
    new_labels = label_session(
        items,
        already_labeled={},
        input_fn=scripted_input(["not-a-label", "partial"]),
        print_fn=lambda s: None,
    )
    assert new_labels[0]["human_label"] == "partial"


def test_label_session_blank_skips_item() -> None:
    items = [make_question("q1"), make_question("q2")]
    new_labels = label_session(
        items, already_labeled={}, input_fn=scripted_input(["", "correct"]), print_fn=lambda s: None
    )
    assert [r["id"] for r in new_labels] == ["q2"]


def test_label_session_quit_stops_early() -> None:
    items = [make_question("q1"), make_question("q2"), make_question("q3")]
    new_labels = label_session(
        items,
        already_labeled={},
        input_fn=scripted_input(["correct", "quit"]),
        print_fn=lambda s: None,
    )
    assert [r["id"] for r in new_labels] == ["q1"]


def test_label_session_skips_already_labeled_items() -> None:
    items = [make_question("q1"), make_question("q2")]
    already_labeled = {"q1": {"id": "q1", "human_label": "correct"}}
    new_labels = label_session(
        items,
        already_labeled=already_labeled,
        input_fn=scripted_input(["partial"]),
        print_fn=lambda s: None,
    )
    assert [r["id"] for r in new_labels] == ["q2"]


def test_append_and_load_labels_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "labels" / "human_labels.jsonl"
    records = [
        {"id": "q1", "human_label": "correct"},
        {"id": "q2", "human_label": "partial"},
    ]
    append_labels(path, records)

    loaded = load_existing_labels(path)
    assert set(loaded) == {"q1", "q2"}
    assert loaded["q1"]["human_label"] == "correct"


def test_append_labels_is_additive_across_calls(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    append_labels(path, [{"id": "q1", "human_label": "correct"}])
    append_labels(path, [{"id": "q2", "human_label": "incorrect"}])
    loaded = load_existing_labels(path)
    assert set(loaded) == {"q1", "q2"}


def test_load_existing_labels_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_existing_labels(tmp_path / "nonexistent.jsonl") == {}
