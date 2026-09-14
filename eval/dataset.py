"""Golden-set loading for the evaluation harness.

Distinct from ``tests/test_golden_set.py`` (which guards the benchmark file's
schema/integrity as a static asset): this module is the typed loader the
runtime evaluation pipeline (``eval/run.py``) actually consumes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

REFUSAL_REQUIRED_TYPES = ("unanswerable", "false_premise")


class DatasetError(RuntimeError):
    """Raised when the golden set file is missing or malformed."""


@dataclass(frozen=True)
class GoldenQuestion:
    """One golden-set record, typed for the evaluation runner."""

    id: str
    type: str
    question: str
    reference_answer: str
    supporting_chunk_ids: list[str]
    tags: list[str]

    @property
    def requires_refusal(self) -> bool:
        return self.type in REFUSAL_REQUIRED_TYPES


def load_golden_set(path: str | Path) -> list[GoldenQuestion]:
    """Load and parse every record in a golden-set JSONL file.

    Raises :class:`DatasetError` on a missing file, malformed JSON, or a
    record missing a required field - failing loudly here is preferable to
    an evaluation run silently skipping or misreading questions.
    """
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"Golden set not found at {path}")

    questions: list[GoldenQuestion] = []
    with path.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path.name}:{line_num}: malformed JSON: {exc}") from exc
            try:
                questions.append(
                    GoldenQuestion(
                        id=raw["id"],
                        type=raw["type"],
                        question=raw["question"],
                        reference_answer=raw["reference_answer"],
                        supporting_chunk_ids=list(raw["supporting_chunk_ids"]),
                        tags=list(raw["tags"]),
                    )
                )
            except KeyError as exc:
                raise DatasetError(f"{path.name}:{line_num}: missing field {exc}") from exc

    if not questions:
        raise DatasetError(f"Golden set at {path} contained no records")
    return questions
