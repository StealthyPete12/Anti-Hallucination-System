"""Unit tests for eval/metrics/refusal.py.

Uses a FakeJudge (satisfying the same ``async evaluate(prompt_name, variables)``
shape as eval.judge.LLMJudge) so these tests never touch litellm - only
eval/judge.py's own tests exercise the real LiteLLM call path.
"""

from __future__ import annotations

from typing import Any

import pytest

from eval.judge import JudgeResponse
from eval.metrics.refusal import (
    REFUSAL_REQUIRED_TYPES,
    RefusalEvaluation,
    evaluate_refusal,
    fabrication_count,
    refusal_rate,
)
from rag.models import TokenUsage

pytestmark = pytest.mark.anyio


def make_response(data: dict[str, Any]) -> JudgeResponse:
    return JudgeResponse(
        data=data,
        prompt_name="refusal_v1",
        prompt_version="v1",
        prompt_hash="deadbeef",
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        cost_usd=0.0001,
        latency_ms=12.3,
        raw_content=str(data),
        attempts=1,
    )


class FakeJudge:
    def __init__(self, response: JudgeResponse) -> None:
        self._response = response
        self.calls: list[tuple[str, dict[str, str]]] = []

    async def evaluate(self, prompt_name: str, variables: dict[str, str]) -> JudgeResponse:
        self.calls.append((prompt_name, variables))
        return self._response


# ---------------------------------------------------------------------------
# REFUSAL_REQUIRED_TYPES
# ---------------------------------------------------------------------------


def test_refusal_required_types() -> None:
    assert REFUSAL_REQUIRED_TYPES == ("unanswerable", "false_premise")


# ---------------------------------------------------------------------------
# evaluate_refusal
# ---------------------------------------------------------------------------


async def test_evaluate_refusal_wires_all_fields() -> None:
    judge = FakeJudge(make_response({"refused": True, "reason": "correctly declined"}))
    result = await evaluate_refusal(
        judge,
        question="What is the max row count?",
        question_type="unanswerable",
        reference_answer="The context does not contain enough information.",
        generated_answer=(
            "The supplied context does not contain enough information to answer that question."
        ),
    )

    assert result.refused is True
    assert result.reason == "correctly declined"
    assert judge.calls[0][0] == "refusal_v1"
    assert judge.calls[0][1]["question_type"] == "unanswerable"


async def test_evaluate_refusal_false_when_system_fabricated() -> None:
    judge = FakeJudge(make_response({"refused": False, "reason": "answered anyway"}))
    result = await evaluate_refusal(
        judge,
        question="q",
        question_type="false_premise",
        reference_answer="ref",
        generated_answer="gen",
    )
    assert result.refused is False


async def test_evaluate_refusal_missing_field_raises() -> None:
    judge = FakeJudge(make_response({"reason": "no refused field"}))
    with pytest.raises(ValueError):
        await evaluate_refusal(
            judge,
            question="q",
            question_type="unanswerable",
            reference_answer="ref",
            generated_answer="gen",
        )


async def test_evaluate_refusal_missing_reason_defaults_to_empty_string() -> None:
    judge = FakeJudge(make_response({"refused": True}))
    result = await evaluate_refusal(
        judge,
        question="q",
        question_type="unanswerable",
        reference_answer="ref",
        generated_answer="gen",
    )
    assert result.reason == ""


# ---------------------------------------------------------------------------
# refusal_rate
# ---------------------------------------------------------------------------


def test_refusal_rate_all_refused() -> None:
    evaluations = [
        RefusalEvaluation(refused=True, reason="", judge_response=make_response({})),
        RefusalEvaluation(refused=True, reason="", judge_response=make_response({})),
    ]
    assert refusal_rate(evaluations) == 1.0


def test_refusal_rate_none_refused() -> None:
    evaluations = [
        RefusalEvaluation(refused=False, reason="", judge_response=make_response({})),
        RefusalEvaluation(refused=False, reason="", judge_response=make_response({})),
    ]
    assert refusal_rate(evaluations) == 0.0


def test_refusal_rate_mixed() -> None:
    evaluations = [
        RefusalEvaluation(refused=True, reason="", judge_response=make_response({})),
        RefusalEvaluation(refused=False, reason="", judge_response=make_response({})),
        RefusalEvaluation(refused=True, reason="", judge_response=make_response({})),
        RefusalEvaluation(refused=False, reason="", judge_response=make_response({})),
    ]
    assert refusal_rate(evaluations) == pytest.approx(0.5)


def test_refusal_rate_empty_list_is_none() -> None:
    assert refusal_rate([]) is None


# ---------------------------------------------------------------------------
# fabrication_count
# ---------------------------------------------------------------------------


def test_fabrication_count_counts_unrefused() -> None:
    evaluations = [
        RefusalEvaluation(refused=True, reason="", judge_response=make_response({})),
        RefusalEvaluation(refused=False, reason="", judge_response=make_response({})),
        RefusalEvaluation(refused=False, reason="", judge_response=make_response({})),
    ]
    assert fabrication_count(evaluations) == 2


def test_fabrication_count_zero_when_all_refused() -> None:
    evaluations = [
        RefusalEvaluation(refused=True, reason="", judge_response=make_response({})),
    ]
    assert fabrication_count(evaluations) == 0


def test_fabrication_count_empty_list_is_zero() -> None:
    assert fabrication_count([]) == 0
