"""Unit tests for rag/cost.py (Phase 5 config-driven cost calculation)."""

from __future__ import annotations

from rag.cost import estimate_cost_usd


def test_known_model_computes_cost_from_config_rates() -> None:
    # gpt-4o-mini: input_per_1k=0.00015, output_per_1k=0.0006 (see config.MODEL_RATES)
    cost = estimate_cost_usd("gpt-4o-mini", prompt_tokens=1000, completion_tokens=1000)
    assert cost == 0.00015 + 0.0006


def test_unknown_model_returns_none() -> None:
    assert estimate_cost_usd("some-unlisted-model", prompt_tokens=100, completion_tokens=50) is None


def test_zero_tokens_is_zero_cost() -> None:
    assert estimate_cost_usd("gpt-4o-mini", prompt_tokens=0, completion_tokens=0) == 0.0
