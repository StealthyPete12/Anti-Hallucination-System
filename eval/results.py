"""The result of evaluating one golden-set question end to end.

Kept as a plain, JSON-serializable dataclass (rather than nesting the richer
``eval.metrics.*`` dataclasses directly) so a scorecard's ``per_question``
entries are exactly what gets written to disk, with no separate translation
step to get wrong.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class QuestionResult:
    """One golden-set question's full evaluation record."""

    id: str
    type: str
    question: str
    generated_answer: str
    reference_answer: str
    retrieved_chunk_ids: list[str]
    retrieval_metrics: dict[str, float | None]
    faithfulness: dict[str, Any] | None
    correctness: dict[str, Any] | None
    refusal: dict[str, Any] | None
    latency_ms: dict[str, float]
    cost_usd: float
    trace_id: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "question": self.question,
            "generated_answer": self.generated_answer,
            "reference_answer": self.reference_answer,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "retrieval_metrics": self.retrieval_metrics,
            "faithfulness": self.faithfulness,
            "correctness": self.correctness,
            "refusal": self.refusal,
            "latency_ms": self.latency_ms,
            "cost_usd": self.cost_usd,
            "trace_id": self.trace_id,
            "error": self.error,
        }
