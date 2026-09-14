"""Unit tests for LiteLLMQueryRewriter (rag/query_rewriter.py).

litellm.completion is monkeypatched, same pattern as tests/unit/test_generator.py.
"""

from __future__ import annotations

from typing import Any

import pytest
from litellm.types.utils import Choices, Message, ModelResponse, Usage

from rag.query_rewriter import LiteLLMQueryRewriter


def fake_response(content: str) -> ModelResponse:
    return ModelResponse(
        choices=[Choices(message=Message(content=content, role="assistant"))],
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    )


def test_rewrite_returns_model_output(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_completion(**kwargs: Any) -> ModelResponse:
        return fake_response("What is PostgreSQL's default transaction isolation level?")

    monkeypatch.setattr("rag.query_rewriter.litellm.completion", fake_completion)

    rewriter = LiteLLMQueryRewriter(model="fake-model")
    result = rewriter.rewrite("what's the default for that isolation thing?")

    assert result == "What is PostgreSQL's default transaction isolation level?"


def test_rewrite_strips_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "rag.query_rewriter.litellm.completion",
        lambda **kwargs: fake_response("  rewritten query  \n"),
    )
    rewriter = LiteLLMQueryRewriter(model="fake-model")
    assert rewriter.rewrite("original") == "rewritten query"


def test_rewrite_falls_back_to_original_on_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raising_completion(**kwargs: Any) -> ModelResponse:
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr("rag.query_rewriter.litellm.completion", raising_completion)

    rewriter = LiteLLMQueryRewriter(model="fake-model")
    result = rewriter.rewrite("original question")

    assert result == "original question"


def test_rewrite_falls_back_to_original_on_empty_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "rag.query_rewriter.litellm.completion", lambda **kwargs: fake_response("   ")
    )
    rewriter = LiteLLMQueryRewriter(model="fake-model")
    assert rewriter.rewrite("original question") == "original question"


def test_rewrite_passes_model_and_generation_params(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> ModelResponse:
        captured.update(kwargs)
        return fake_response("rewritten")

    monkeypatch.setattr("rag.query_rewriter.litellm.completion", fake_completion)

    rewriter = LiteLLMQueryRewriter(
        model="gpt-4o-mini", temperature=0.0, max_tokens=64, timeout=5.0
    )
    rewriter.rewrite("question")

    assert captured["model"] == "gpt-4o-mini"
    assert captured["temperature"] == 0.0
    assert captured["max_tokens"] == 64
    assert captured["timeout"] == 5.0
