"""Scorecard aggregation: turns a list of per-question results into the run's
overall retrieval/generation/safety/ops summary, and turns several scorecards
(one per ``--repeats`` pass) into a noise-floor report.

Pure aggregation, no I/O and no LLM calls, so it can be unit tested with
hand-built ``QuestionResult`` lists independent of the async runner.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Any

from eval.results import QuestionResult

#: (scorecard section, metric key) pairs the noise floor is computed over.
NOISE_FLOOR_METRICS: tuple[tuple[str, str], ...] = (
    ("retrieval", "recall_at_5"),
    ("retrieval", "mrr"),
    ("retrieval", "ndcg_at_10"),
    ("generation", "faithfulness"),
    ("generation", "correctness"),
    ("safety", "refusal_rate"),
    ("ops", "mean_cost_usd"),
    ("ops", "p95_latency_ms"),
)


def _mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    if not present:
        return None
    return sum(present) / len(present)


def _percentile(values: list[float], pct: float) -> float | None:
    """Linear-interpolation percentile (same method as numpy's default)."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def build_scorecard(
    results: list[QuestionResult],
    *,
    git_sha: str,
    timestamp: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate one evaluation pass's per-question results into a scorecard.

    Questions that errored out (``result.error is not None``) are excluded
    from every numeric aggregate but still appear in ``per_question`` for
    debugging - a crashed question should never silently inflate a score by
    disappearing from the denominator undetected, so ``num_errors`` is
    surfaced at the top level instead.
    """
    ok = [r for r in results if r.error is None]

    recall_values = [r.retrieval_metrics.get("recall_at_5") for r in ok]
    mrr_values = [r.retrieval_metrics.get("mrr") for r in ok]
    ndcg_values = [r.retrieval_metrics.get("ndcg_at_10") for r in ok]

    faithfulness_values = [
        r.faithfulness["score"]
        for r in ok
        if r.faithfulness is not None and r.faithfulness.get("score") is not None
    ]
    correctness_values = [r.correctness["score"] for r in ok if r.correctness is not None]

    refusal_entries = [r.refusal for r in ok if r.refusal is not None]
    refused_count = sum(1 for entry in refusal_entries if entry["refused"])
    refusal_rate = refused_count / len(refusal_entries) if refusal_entries else None
    fabrications = sum(1 for entry in refusal_entries if not entry["refused"])

    cost_values = [r.cost_usd for r in ok]
    latency_values = [r.latency_ms["total"] for r in ok if "total" in r.latency_ms]

    return {
        "git_sha": git_sha,
        "timestamp": timestamp,
        "config": config,
        "retrieval": {
            "recall_at_5": _mean(recall_values),
            "mrr": _mean(mrr_values),
            "ndcg_at_10": _mean(ndcg_values),
        },
        "generation": {
            "faithfulness": _mean(faithfulness_values),
            "correctness": _mean(correctness_values),
        },
        "safety": {
            "refusal_rate": refusal_rate,
            "fabrications": fabrications,
        },
        "ops": {
            "mean_cost_usd": _mean(cost_values),
            "p95_latency_ms": _percentile(latency_values, 0.95),
        },
        "num_questions": len(results),
        "num_errors": len(results) - len(ok),
        "per_question": [r.to_dict() for r in results],
    }


def compute_noise_floor(scorecards: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    """Mean and standard deviation of each headline metric across repeated runs.

    This is the project's noise floor: a future improvement's measured delta
    on a metric must exceed this run-to-run variance before it can be
    trusted as a real change rather than evaluation noise. Requires at least
    2 runs to compute a standard deviation; a single value's stdev is
    reported as 0.0 (no variance observed, not "no variance exists").
    """
    noise_floor: dict[str, dict[str, float | None]] = {}
    for section, key in NOISE_FLOOR_METRICS:
        values = [sc[section][key] for sc in scorecards if sc.get(section, {}).get(key) is not None]
        metric_name = f"{section}.{key}"
        if not values:
            noise_floor[metric_name] = {"mean": None, "stdev": None, "n": 0}
        elif len(values) == 1:
            noise_floor[metric_name] = {"mean": values[0], "stdev": 0.0, "n": 1}
        else:
            noise_floor[metric_name] = {
                "mean": statistics.mean(values),
                "stdev": statistics.stdev(values),
                "n": len(values),
            }
    return noise_floor
