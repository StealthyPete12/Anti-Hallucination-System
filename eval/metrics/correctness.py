"""Correctness metric: answer accuracy relative to the golden set's reference answer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from eval.judge import JudgeResponse, LLMJudge

PROMPT_NAME = "correctness_v1"

VALID_VERDICTS = ("correct", "partial", "incorrect")
VERDICT_SCORES: dict[str, float] = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


@dataclass(frozen=True)
class CorrectnessEvaluation:
    verdict: str
    reason: str
    score: float
    judge_response: JudgeResponse


def parse_verdict(data: dict[str, Any]) -> tuple[str, str]:
    verdict = data.get("verdict")
    if verdict not in VALID_VERDICTS:
        raise ValueError(
            f"Correctness judge returned invalid verdict {verdict!r}, expected one of "
            f"{VALID_VERDICTS}: {data!r}"
        )
    reason = str(data.get("reason", ""))
    return verdict, reason


async def evaluate_correctness(
    judge: LLMJudge, *, question: str, reference_answer: str, generated_answer: str
) -> CorrectnessEvaluation:
    response = await judge.evaluate(
        PROMPT_NAME,
        {
            "question": question,
            "reference_answer": reference_answer,
            "generated_answer": generated_answer,
        },
    )
    verdict, reason = parse_verdict(response.data)
    return CorrectnessEvaluation(
        verdict=verdict,
        reason=reason,
        score=VERDICT_SCORES[verdict],
        judge_response=response,
    )


def mean_correctness(evaluations: list[CorrectnessEvaluation]) -> float | None:
    if not evaluations:
        return None
    return sum(e.score for e in evaluations) / len(evaluations)
