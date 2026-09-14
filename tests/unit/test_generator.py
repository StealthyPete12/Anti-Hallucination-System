"""Unit tests for LiteLLMGenerator's output contract (rag/generator.py).

litellm.completion is monkeypatched so these tests never make a real network
call to an LLM provider - they exercise the citation-extraction and
usage-parsing contract in isolation.
"""

from __future__ import annotations

from typing import Any

import pytest
from litellm.types.utils import Choices, Message, ModelResponse, Usage

from rag.generator import GenerationError, LiteLLMGenerator
from rag.models import RetrievedChunk


def make_chunk(chunk_id: str, text: str = "some passage text") -> RetrievedChunk:
    doc_id = chunk_id.split("#")[0]
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        score=0.9,
        source_path=f"data/corpus/{doc_id}.txt",
        heading="H",
        chunk_index=0,
    )


def fake_response(content: str, usage: Usage | None = None) -> ModelResponse:
    return ModelResponse(
        choices=[Choices(message=Message(content=content, role="assistant"))],
        usage=usage or Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
    )


def test_valid_citations_are_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_completion(**kwargs: Any) -> ModelResponse:
        return fake_response("The retention period is 90 days [ops-guide#14].")

    monkeypatch.setattr("rag.generator.litellm.completion", fake_completion)

    generator = LiteLLMGenerator(model="fake-model")
    chunks = [make_chunk("ops-guide#14")]
    result = generator.generate("What is the retention period?", chunks)

    assert result.citations == ["ops-guide#14"]
    assert "90 days" in result.answer


def test_fabricated_citations_are_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_completion(**kwargs: Any) -> ModelResponse:
        return fake_response("The answer is X [ops-guide#14] and also [made-up#99].")

    monkeypatch.setattr("rag.generator.litellm.completion", fake_completion)

    generator = LiteLLMGenerator(model="fake-model")
    chunks = [make_chunk("ops-guide#14")]
    result = generator.generate("question", chunks)

    assert result.citations == ["ops-guide#14"]
    assert "made-up#99" not in result.citations


def test_duplicate_citations_are_deduplicated_preserving_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_completion(**kwargs: Any) -> ModelResponse:
        return fake_response("[a#0] then again [b#1] and once more [a#0].")

    monkeypatch.setattr("rag.generator.litellm.completion", fake_completion)

    generator = LiteLLMGenerator(model="fake-model")
    chunks = [make_chunk("a#0"), make_chunk("b#1")]
    result = generator.generate("question", chunks)

    assert result.citations == ["a#0", "b#1"]


def test_refusal_answer_has_no_citations(monkeypatch: pytest.MonkeyPatch) -> None:
    refusal = "The supplied context does not contain enough information to answer that question."

    def fake_completion(**kwargs: Any) -> ModelResponse:
        return fake_response(refusal)

    monkeypatch.setattr("rag.generator.litellm.completion", fake_completion)

    generator = LiteLLMGenerator(model="fake-model")
    result = generator.generate("unanswerable question", [])

    assert result.answer == refusal
    assert result.citations == []


def test_usage_is_extracted_from_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_completion(**kwargs: Any) -> ModelResponse:
        return fake_response(
            "answer [a#0]", usage=Usage(prompt_tokens=50, completion_tokens=10, total_tokens=60)
        )

    monkeypatch.setattr("rag.generator.litellm.completion", fake_completion)

    generator = LiteLLMGenerator(model="fake-model")
    result = generator.generate("q", [make_chunk("a#0")])

    assert result.usage.prompt_tokens == 50
    assert result.usage.completion_tokens == 10
    assert result.usage.total_tokens == 60


def test_cost_is_extracted_from_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_completion(**kwargs: Any) -> ModelResponse:
        return fake_response("answer [a#0]")

    monkeypatch.setattr("rag.generator.litellm.completion", fake_completion)
    monkeypatch.setattr("rag.generator.litellm.completion_cost", lambda **kwargs: 0.00123)

    generator = LiteLLMGenerator(model="fake-model")
    result = generator.generate("q", [make_chunk("a#0")])

    assert result.cost_usd == pytest.approx(0.00123)


def test_cost_defaults_to_zero_when_lookup_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_completion(**kwargs: Any) -> ModelResponse:
        return fake_response("answer [a#0]")

    def failing_cost(**kwargs: Any) -> float:
        raise ValueError("unpriced model")

    monkeypatch.setattr("rag.generator.litellm.completion", fake_completion)
    monkeypatch.setattr("rag.generator.litellm.completion_cost", failing_cost)

    generator = LiteLLMGenerator(model="fake-model")
    result = generator.generate("q", [make_chunk("a#0")])

    assert result.cost_usd == 0.0


def test_llm_call_failure_raises_generation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def failing_completion(**kwargs: Any) -> ModelResponse:
        raise RuntimeError("provider unreachable")

    monkeypatch.setattr("rag.generator.litellm.completion", failing_completion)

    generator = LiteLLMGenerator(model="fake-model")
    with pytest.raises(GenerationError):
        generator.generate("q", [make_chunk("a#0")])


def test_system_prompt_and_context_are_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> ModelResponse:
        captured.update(kwargs)
        return fake_response("answer [a#0]")

    monkeypatch.setattr("rag.generator.litellm.completion", fake_completion)

    generator = LiteLLMGenerator(model="fake-model", temperature=0.2, max_tokens=500)
    chunk = make_chunk("a#0", text="the retention period is 90 days")
    generator.generate("What is the retention period?", [chunk])

    messages = captured["messages"]
    assert messages[0]["role"] == "system"
    assert "refuse" in messages[0]["content"].lower()
    assert "[a#0]" in messages[1]["content"]
    assert "the retention period is 90 days" in messages[1]["content"]
    assert captured["temperature"] == 0.2
    assert captured["max_tokens"] == 500
