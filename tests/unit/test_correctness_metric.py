"""Unit tests for eval/metrics/correctness.py.

Uses a FakeJudge (satisfying the same ``async evaluate(prompt_name, variables)``
shape as eval.judge.LLMJudge) so these tests never touch litellm - only
eval/judge.py's own tests exercise the real LiteLLM call path.
"""

from __future__ import annotations

from typing import Any

import pytest

from eval.judge import JudgeResponse
from eval.metrics.correctness import (
    VERDICT_SCORES,
    CorrectnessEvaluation,
    evaluate_correctness,
    mean_correctness,
    parse_verdict,
)
from rag.models import TokenUsage

pytestmark = pytest.mark.anyio


def make_response(data: dict[str, Any]) -> JudgeResponse:
    return JudgeResponse(
        data=data,
        prompt_name="correctness_v1",
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
# parse_verdict
# ---------------------------------------------------------------------------


def test_parse_verdict_correct() -> None:
    verdict, reason = parse_verdict({"verdict": "correct", "reason": "matches"})
    assert verdict == "correct"
    assert reason == "matches"


def test_parse_verdict_partial() -> None:
    verdict, _ = parse_verdict({"verdict": "partial", "reason": "incomplete"})
    assert verdict == "partial"


def test_parse_verdict_incorrect() -> None:
    verdict, _ = parse_verdict({"verdict": "incorrect", "reason": "wrong"})
    assert verdict == "incorrect"


def test_parse_verdict_missing_reason_defaults_to_empty_string() -> None:
    _, reason = parse_verdict({"verdict": "correct"})
    assert reason == ""


def test_parse_verdict_invalid_value_raises() -> None:
    with pytest.raises(ValueError):
        parse_verdict({"verdict": "sort-of", "reason": "??"})


def test_parse_verdict_missing_key_raises() -> None:
    with pytest.raises(ValueError):
        parse_verdict({"reason": "no verdict field"})


# ---------------------------------------------------------------------------
# VERDICT_SCORES
# ---------------------------------------------------------------------------


def test_verdict_scores_mapping() -> None:
    assert VERDICT_SCORES == {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


# ---------------------------------------------------------------------------
# evaluate_correctness
# ---------------------------------------------------------------------------


async def test_evaluate_correctness_wires_question_and_answers() -> None:
    judge = FakeJudge(make_response({"verdict": "correct", "reason": "matches reference"}))
    result = await evaluate_correctness(
        judge,
        question="What is MVCC?",
        reference_answer="Multiversion Concurrency Control.",
        generated_answer="MVCC stands for Multiversion Concurrency Control.",
    )

    assert result.verdict == "correct"
    assert result.reason == "matches reference"
    assert result.score == 1.0
    assert judge.calls[0][0] == "correctness_v1"
    assert judge.calls[0][1]["question"] == "What is MVCC?"
    assert judge.calls[0][1]["reference_answer"] == "Multiversion Concurrency Control."
    assert (
        judge.calls[0][1]["generated_answer"] == "MVCC stands for Multiversion Concurrency Control."
    )


async def test_evaluate_correctness_partial_scores_half() -> None:
    judge = FakeJudge(make_response({"verdict": "partial", "reason": "missing detail"}))
    result = await evaluate_correctness(
        judge, question="q", reference_answer="ref", generated_answer="gen"
    )
    assert result.score == 0.5


async def test_evaluate_correctness_incorrect_scores_zero() -> None:
    judge = FakeJudge(make_response({"verdict": "incorrect", "reason": "wrong"}))
    result = await evaluate_correctness(
        judge, question="q", reference_answer="ref", generated_answer="gen"
    )
    assert result.score == 0.0


async def test_evaluate_correctness_invalid_verdict_raises() -> None:
    judge = FakeJudge(make_response({"verdict": "maybe", "reason": "??"}))
    with pytest.raises(ValueError):
        await evaluate_correctness(
            judge, question="q", reference_answer="ref", generated_answer="gen"
        )


# ---------------------------------------------------------------------------
# mean_correctness
# ---------------------------------------------------------------------------


def test_mean_correctness_averages_scores() -> None:
    evaluations = [
        CorrectnessEvaluation(
            verdict="correct", reason="", score=1.0, judge_response=make_response({})
        ),
        CorrectnessEvaluation(
            verdict="partial", reason="", score=0.5, judge_response=make_response({})
        ),
        CorrectnessEvaluation(
            verdict="incorrect", reason="", score=0.0, judge_response=make_response({})
        ),
    ]
    assert mean_correctness(evaluations) == pytest.approx(0.5)


def test_mean_correctness_empty_list_is_none() -> None:
    assert mean_correctness([]) is None


def test_mean_correctness_all_correct_is_one() -> None:
    evaluations = [
        CorrectnessEvaluation(
            verdict="correct", reason="", score=1.0, judge_response=make_response({})
        ),
        CorrectnessEvaluation(
            verdict="correct", reason="", score=1.0, judge_response=make_response({})
        ),
    ]
    assert mean_correctness(evaluations) == pytest.approx(1.0)
