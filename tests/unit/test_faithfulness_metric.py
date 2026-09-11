"""Unit tests for eval/metrics/faithfulness.py.

Uses a FakeJudge (satisfying the same ``async evaluate(prompt_name, variables)``
shape as eval.judge.LLMJudge) so these tests never touch litellm - only
eval/judge.py's own tests exercise the real LiteLLM call path.
"""

from __future__ import annotations

from typing import Any

import pytest

from eval.judge import JudgeResponse
from eval.metrics.faithfulness import (
    Claim,
    evaluate_faithfulness,
    format_context,
    mean_faithfulness,
    parse_claims,
    score_claims,
)
from rag.models import TokenUsage

pytestmark = pytest.mark.anyio


def make_response(data: dict[str, Any]) -> JudgeResponse:
    return JudgeResponse(
        data=data,
        prompt_name="faithfulness_v1",
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
# format_context
# ---------------------------------------------------------------------------


def test_format_context_empty() -> None:
    assert format_context([]) == "(no context retrieved)"


def test_format_context_joins_bracketed_chunk_ids() -> None:
    result = format_context([("a#0", "text one"), ("b#1", "text two")])
    assert result == "[a#0]\ntext one\n\n[b#1]\ntext two"


# ---------------------------------------------------------------------------
# parse_claims
# ---------------------------------------------------------------------------


def test_parse_claims_happy_path() -> None:
    claims, unsupported = parse_claims(
        {
            "claims": [
                {"text": "MVCC stands for Multiversion Concurrency Control.", "supported": True, "chunk_id": "a#0"},
                {"text": "It was invented in 2020.", "supported": False, "chunk_id": None},
            ],
            "unsupported_count": 1,
        }
    )
    assert claims == [
        Claim(text="MVCC stands for Multiversion Concurrency Control.", supported=True, chunk_id="a#0"),
        Claim(text="It was invented in 2020.", supported=False, chunk_id=None),
    ]
    assert unsupported == 1


def test_parse_claims_computes_unsupported_count_when_missing() -> None:
    claims, unsupported = parse_claims(
        {
            "claims": [
                {"text": "x", "supported": True, "chunk_id": "a#0"},
                {"text": "y", "supported": False, "chunk_id": None},
                {"text": "z", "supported": False, "chunk_id": None},
            ]
        }
    )
    assert len(claims) == 3
    assert unsupported == 2


def test_parse_claims_empty_list_for_pure_refusal() -> None:
    claims, unsupported = parse_claims({"claims": [], "unsupported_count": 0})
    assert claims == []
    assert unsupported == 0


def test_parse_claims_missing_claims_key_raises() -> None:
    with pytest.raises(ValueError):
        parse_claims({"unsupported_count": 0})


def test_parse_claims_malformed_claim_raises() -> None:
    with pytest.raises(ValueError):
        parse_claims({"claims": [{"text": "x"}]})  # missing "supported"


# ---------------------------------------------------------------------------
# score_claims
# ---------------------------------------------------------------------------


def test_score_claims_all_supported() -> None:
    claims = [Claim("a", True, "c#0"), Claim("b", True, "c#1")]
    assert score_claims(claims) == 1.0


def test_score_claims_partial_support() -> None:
    # 8 supported, 10 total -> 0.80, matching the worked example in the brief.
    claims = [Claim(f"c{i}", i < 8, "x#0") for i in range(10)]
    assert score_claims(claims) == pytest.approx(0.80)


def test_score_claims_none_supported() -> None:
    claims = [Claim("a", False, None), Claim("b", False, None)]
    assert score_claims(claims) == 0.0


def test_score_claims_empty_is_vacuously_faithful() -> None:
    assert score_claims([]) == 1.0


# ---------------------------------------------------------------------------
# evaluate_faithfulness
# ---------------------------------------------------------------------------


async def test_evaluate_faithfulness_wires_context_and_answer() -> None:
    judge = FakeJudge(
        make_response(
            {
                "claims": [{"text": "MVCC.", "supported": True, "chunk_id": "a#0"}],
                "unsupported_count": 0,
            }
        )
    )
    result = await evaluate_faithfulness(
        judge, answer="MVCC.", retrieved_chunks=[("a#0", "Multiversion Concurrency Control.")]
    )

    assert result.score == 1.0
    assert result.unsupported_count == 0
    assert result.claims[0].chunk_id == "a#0"
    assert judge.calls[0][0] == "faithfulness_v1"
    assert "Multiversion Concurrency Control." in judge.calls[0][1]["context"]
    assert judge.calls[0][1]["answer"] == "MVCC."


# ---------------------------------------------------------------------------
# mean_faithfulness
# ---------------------------------------------------------------------------


def test_mean_faithfulness_averages_scores() -> None:
    from eval.metrics.faithfulness import FaithfulnessEvaluation

    evaluations = [
        FaithfulnessEvaluation(claims=[], unsupported_count=0, score=1.0, judge_response=make_response({})),
        FaithfulnessEvaluation(claims=[], unsupported_count=0, score=0.5, judge_response=make_response({})),
    ]
    assert mean_faithfulness(evaluations) == pytest.approx(0.75)


def test_mean_faithfulness_empty_list_is_none() -> None:
    assert mean_faithfulness([]) is None


def test_mean_faithfulness_ignores_none_scores() -> None:
    from eval.metrics.faithfulness import FaithfulnessEvaluation

    evaluations = [
        FaithfulnessEvaluation(claims=[], unsupported_count=0, score=None, judge_response=make_response({})),
        FaithfulnessEvaluation(claims=[], unsupported_count=0, score=0.6, judge_response=make_response({})),
    ]
    assert mean_faithfulness(evaluations) == pytest.approx(0.6)
