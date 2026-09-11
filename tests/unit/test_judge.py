"""Unit tests for eval/judge.py: prompt loading/hashing and the LiteLLM judge client.

litellm.acompletion is monkeypatched so these tests never make a real network
call - they exercise prompt rendering, retry/backoff, JSON parsing, and
cost/usage extraction in isolation, mirroring tests/unit/test_generator.py's
approach for the production generator.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
from litellm.types.utils import Choices, Message, ModelResponse, Usage

from eval.judge import JudgeError, LLMJudge, load_prompt

pytestmark = pytest.mark.anyio


def fake_response(content: str, usage: Usage | None = None) -> ModelResponse:
    return ModelResponse(
        choices=[Choices(message=Message(content=content, role="assistant"))],
        usage=usage or Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
    )


# ---------------------------------------------------------------------------
# Prompt loading and hashing
# ---------------------------------------------------------------------------


def test_load_prompt_reads_file_and_hashes_it() -> None:
    template = load_prompt("faithfulness_v1")
    expected_hash = hashlib.sha256(template.text.encode("utf-8")).hexdigest()
    assert template.sha256 == expected_hash
    assert template.version == "v1"
    assert template.name == "faithfulness_v1"
    assert "$context" in template.text
    assert "$answer" in template.text


def test_load_prompt_is_cached() -> None:
    assert load_prompt("correctness_v1") is load_prompt("correctness_v1")


def test_load_prompt_unknown_name_raises_judge_error() -> None:
    with pytest.raises(JudgeError):
        load_prompt("does_not_exist_v1")


def test_all_versioned_prompts_load_and_hash_distinctly() -> None:
    names = ["faithfulness_v1", "correctness_v1", "refusal_v1"]
    hashes = {load_prompt(name).sha256 for name in names}
    assert len(hashes) == len(names)


# ---------------------------------------------------------------------------
# LLMJudge.evaluate
# ---------------------------------------------------------------------------


async def test_evaluate_happy_path_parses_json_and_records_prompt_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_acompletion(**kwargs: Any) -> ModelResponse:
        return fake_response('{"verdict": "correct", "reason": "matches"}')

    monkeypatch.setattr("eval.judge.litellm.acompletion", fake_acompletion)
    monkeypatch.setattr("eval.judge.litellm.completion_cost", lambda **kwargs: 0.001234)

    judge = LLMJudge(model="fake-model")
    response = await judge.evaluate(
        "correctness_v1",
        {"question": "q", "reference_answer": "ref", "generated_answer": "gen"},
    )

    assert response.data == {"verdict": "correct", "reason": "matches"}
    assert response.prompt_name == "correctness_v1"
    assert response.prompt_version == "v1"
    assert response.prompt_hash == load_prompt("correctness_v1").sha256
    assert response.usage.total_tokens == 120
    assert response.cost_usd == pytest.approx(0.001234)
    assert response.attempts == 1


async def test_evaluate_sends_temperature_zero_and_json_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> ModelResponse:
        captured.update(kwargs)
        return fake_response('{"refused": true, "reason": "ok"}')

    monkeypatch.setattr("eval.judge.litellm.acompletion", fake_acompletion)
    monkeypatch.setattr("eval.judge.litellm.completion_cost", lambda **kwargs: 0.0)

    judge = LLMJudge(model="fake-model", temperature=0.0, timeout=45.0)
    await judge.evaluate(
        "refusal_v1",
        {
            "question": "q",
            "question_type": "unanswerable",
            "reference_answer": "ref",
            "generated_answer": "gen",
        },
    )

    assert captured["temperature"] == 0.0
    assert captured["timeout"] == 45.0
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["messages"][0]["role"] == "user"
    assert "q" in captured["messages"][0]["content"]


async def test_evaluate_substitutes_template_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_acompletion(**kwargs: Any) -> ModelResponse:
        captured.update(kwargs)
        return fake_response('{"claims": [], "unsupported_count": 0}')

    monkeypatch.setattr("eval.judge.litellm.acompletion", fake_acompletion)
    monkeypatch.setattr("eval.judge.litellm.completion_cost", lambda **kwargs: 0.0)

    judge = LLMJudge(model="fake-model")
    await judge.evaluate(
        "faithfulness_v1",
        {"context": "[a#0]\nMVCC stands for Multiversion Concurrency Control.", "answer": "MVCC."},
    )

    content = captured["messages"][0]["content"]
    assert "[a#0]" in content
    assert "MVCC stands for Multiversion Concurrency Control." in content
    assert "$context" not in content
    assert "$answer" not in content


async def test_evaluate_missing_template_variable_raises_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def fake_acompletion(**kwargs: Any) -> ModelResponse:
        nonlocal calls
        calls += 1
        return fake_response("{}")

    monkeypatch.setattr("eval.judge.litellm.acompletion", fake_acompletion)

    judge = LLMJudge(model="fake-model")
    with pytest.raises(KeyError):
        await judge.evaluate("correctness_v1", {"question": "q"})  # missing required variables
    assert calls == 0  # never even attempted the LLM call


async def test_evaluate_retries_on_malformed_json_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    async def fake_acompletion(**kwargs: Any) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return fake_response("not valid json")
        return fake_response('{"verdict": "partial", "reason": "close enough"}')

    monkeypatch.setattr("eval.judge.litellm.acompletion", fake_acompletion)
    monkeypatch.setattr("eval.judge.litellm.completion_cost", lambda **kwargs: 0.0)

    judge = LLMJudge(model="fake-model", max_retries=3, retry_base_delay=0.0)
    response = await judge.evaluate(
        "correctness_v1",
        {"question": "q", "reference_answer": "ref", "generated_answer": "gen"},
    )

    assert attempts == 2
    assert response.attempts == 2
    assert response.data["verdict"] == "partial"


async def test_evaluate_raises_judge_error_after_exhausting_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    async def failing_acompletion(**kwargs: Any) -> ModelResponse:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("provider unreachable")

    monkeypatch.setattr("eval.judge.litellm.acompletion", failing_acompletion)

    judge = LLMJudge(model="fake-model", max_retries=3, retry_base_delay=0.0)
    with pytest.raises(JudgeError):
        await judge.evaluate(
            "correctness_v1",
            {"question": "q", "reference_answer": "ref", "generated_answer": "gen"},
        )
    assert attempts == 3


async def test_evaluate_rejects_non_object_json(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_acompletion(**kwargs: Any) -> ModelResponse:
        return fake_response("[1, 2, 3]")

    monkeypatch.setattr("eval.judge.litellm.acompletion", fake_acompletion)

    judge = LLMJudge(model="fake-model", max_retries=1, retry_base_delay=0.0)
    with pytest.raises(JudgeError):
        await judge.evaluate(
            "correctness_v1",
            {"question": "q", "reference_answer": "ref", "generated_answer": "gen"},
        )
