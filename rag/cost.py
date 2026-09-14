"""Per-request cost calculation (Phase 5).

``config.MODEL_RATES`` is the source of truth: when the generation model is
listed there, cost is computed deterministically from token counts and
pricing, no network or SDK call involved. For any other model, callers fall
back to their own estimate (``rag/generator.py`` already had a working
litellm-based fallback from Phase 1 - this module doesn't replace it, only
gives it a config-driven value to prefer when one is available).
"""

from __future__ import annotations

from config import MODEL_RATES


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Compute cost from ``config.MODEL_RATES``, or ``None`` if ``model`` isn't listed."""
    rates = MODEL_RATES.get(model)
    if rates is None:
        return None
    return (prompt_tokens / 1000) * rates["input_per_1k"] + (completion_tokens / 1000) * rates[
        "output_per_1k"
    ]
