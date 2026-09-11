"""Refusal metric: did the system correctly decline an unanswerable question or challenge a
false premise, instead of fabricating an answer?

Only applies to the golden set's ``unanswerable`` and ``false_premise`` question types - the
ones where the *correct* behavior is to NOT answer as asked.
"""

from __future__ import annotations

from dataclasses import dataclass

from eval.judge import JudgeResponse, LLMJudge

PROMPT_NAME = "refusal_v1"

#: Question types the refusal metric applies to.
REFUSAL_REQUIRED_TYPES = ("unanswerable", "false_premise")


@dataclass(frozen=True)
class RefusalEvaluation:
    refused: bool
    reason: str
    judge_response: JudgeResponse


async def evaluate_refusal(
    judge: LLMJudge,
    *,
    question: str,
    question_type: str,
    reference_answer: str,
    generated_answer: str,
) -> RefusalEvaluation:
    response = await judge.evaluate(
        PROMPT_NAME,
        {
            "question": question,
            "question_type": question_type,
            "reference_answer": reference_answer,
            "generated_answer": generated_answer,
        },
    )
    data = response.data
    if "refused" not in data:
        raise ValueError(f"Refusal judge response missing 'refused' field: {data!r}")
    return RefusalEvaluation(
        refused=bool(data["refused"]),
        reason=str(data.get("reason", "")),
        judge_response=response,
    )


def refusal_rate(evaluations: list[RefusalEvaluation]) -> float | None:
    """correct_refusals / required_refusals, over the unanswerable + false_premise subset."""
    if not evaluations:
        return None
    correct = sum(1 for e in evaluations if e.refused)
    return correct / len(evaluations)


def fabrication_count(evaluations: list[RefusalEvaluation]) -> int:
    """Count of unanswerable/false_premise questions the system answered instead of refusing.

    Each one is a fabrication incident: the system asserted something (or
    accepted a false premise) where the correct behavior was to decline.
    """
    return sum(1 for e in evaluations if not e.refused)
