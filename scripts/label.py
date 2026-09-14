"""Judge validation: sample ~40 answers from an evaluation run and manually
label them, so the LLM judge's correctness verdicts can be checked against
a human ground truth (Phase 4 spec, section 11).

    uv run python -m scripts.label --scorecard evals/results.json

Labels persist to a JSONL file (default ``evals/labels/human_labels.jsonl``)
keyed by question id, so re-running the command:

- skips items already labeled in a prior session (safe to Ctrl-C and resume)
- never loses a label to a re-run against an updated scorecard

Each sampling/labeling step is a small, pure(-ish) function taking its I/O
as injectable callables, so the whole workflow is unit-testable without a
real terminal - see ``tests/unit/test_label.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("scripts.label")

DEFAULT_SAMPLE_SIZE = 40
DEFAULT_LABELS_PATH = "evals/labels/human_labels.jsonl"
VALID_LABELS = ("correct", "incorrect", "partial")


def load_per_question(scorecard_path: str | Path) -> list[dict[str, Any]]:
    scorecard = json.loads(Path(scorecard_path).read_text(encoding="utf-8"))
    per_question = scorecard.get("per_question", [])
    return [q for q in per_question if q.get("error") is None]


def load_existing_labels(labels_path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(labels_path)
    if not path.exists():
        return {}
    labels: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            labels[record["id"]] = record
    return labels


def sample_questions(
    per_question: list[dict[str, Any]],
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Deterministically sample up to ``sample_size`` questions (seeded, so the
    sample - and therefore the agreement measurement - is reproducible)."""
    rng = random.Random(seed)
    pool = list(per_question)
    rng.shuffle(pool)
    return pool[:sample_size]


def label_session(
    items: list[dict[str, Any]],
    already_labeled: dict[str, dict[str, Any]],
    *,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> list[dict[str, Any]]:
    """Interactively (or, in tests, via injected ``input_fn``) collect a human
    label for every item in ``items`` not already present in ``already_labeled``.

    A blank/whitespace-only response skips that item without recording
    anything (so a labeler can bail out on an ambiguous case); typing
    "quit" stops the session early. Returns only the newly-collected
    records - callers append them to the persisted labels file.
    """
    new_labels: list[dict[str, Any]] = []
    for item in items:
        qid = item["id"]
        if qid in already_labeled:
            continue

        print_fn(f"\n--- {qid} ({item.get('type', '?')}) ---")
        print_fn(f"Question: {item.get('question', '')}")
        print_fn(f"Reference answer: {item.get('reference_answer', '')}")
        print_fn(f"Generated answer: {item.get('generated_answer', '')}")
        judge_verdict = (item.get("correctness") or {}).get("verdict")
        print_fn(f"[judge verdict: {judge_verdict}]")

        while True:
            response = (
                input_fn(f"Label ({'/'.join(VALID_LABELS)}, blank=skip, quit=stop): ")
                .strip()
                .lower()
            )
            if response == "quit":
                return new_labels
            if response == "":
                break
            if response in VALID_LABELS:
                new_labels.append(
                    {
                        "id": qid,
                        "question": item.get("question", ""),
                        "generated_answer": item.get("generated_answer", ""),
                        "reference_answer": item.get("reference_answer", ""),
                        "judge_verdict": judge_verdict,
                        "human_label": response,
                        "labeled_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    }
                )
                break
            print_fn(f"Unrecognized label {response!r}; try again.")
    return new_labels


def append_labels(labels_path: str | Path, new_labels: list[dict[str, Any]]) -> None:
    if not new_labels:
        return
    path = Path(labels_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in new_labels:
            f.write(json.dumps(record) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample and manually label answers for judge validation (Phase 4)."
    )
    parser.add_argument("--scorecard", default="evals/results.json")
    parser.add_argument("--labels-path", default=DEFAULT_LABELS_PATH)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    per_question = load_per_question(args.scorecard)
    sample = sample_questions(per_question, args.sample_size, args.seed)
    already_labeled = load_existing_labels(args.labels_path)

    remaining = [item for item in sample if item["id"] not in already_labeled]
    num_already_labeled = sum(1 for item in sample if item["id"] in already_labeled)
    print(
        f"Sampled {len(sample)} question(s); {num_already_labeled} already labeled; "
        f"{len(remaining)} remaining."
    )

    new_labels = label_session(sample, already_labeled)
    append_labels(args.labels_path, new_labels)
    print(f"Recorded {len(new_labels)} new label(s) to {args.labels_path}")


if __name__ == "__main__":
    main()
