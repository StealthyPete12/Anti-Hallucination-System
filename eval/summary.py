"""Human-readable summary table rendering for a scorecard (and, if present,
its noise-floor report). Printed to stdout at the end of every ``eval.run``
invocation, independent of the JSON file written alongside it.
"""

from __future__ import annotations

from typing import Any

_RULE = "-" * 54


def _fmt(value: float | int | None, *, suffix: str = "", digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value}{suffix}"
    return f"{value:.{digits}f}{suffix}"


def _fmt_cost(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def _fmt_noise(entry: dict[str, float | None] | None, *, digits: int = 3) -> str:
    if entry is None or entry.get("mean") is None:
        return ""
    mean = entry["mean"]
    stdev = entry.get("stdev") or 0.0
    return f"  (mean={mean:.{digits}f}, stdev={stdev:.{digits}f}, n={entry.get('n', 0)})"


def render_summary(
    scorecard: dict[str, Any], noise_floor: dict[str, dict[str, float | None]] | None = None
) -> str:
    """Render the fixed retrieval/generation/safety/ops summary block."""
    retrieval = scorecard.get("retrieval", {})
    generation = scorecard.get("generation", {})
    safety = scorecard.get("safety", {})
    ops = scorecard.get("ops", {})
    nf = noise_floor or scorecard.get("noise_floor")

    def noise(section: str, key: str) -> str:
        return _fmt_noise(nf.get(f"{section}.{key}")) if nf else ""

    lines = [
        _RULE,
        "Retrieval",
        _RULE,
        f"Recall@5      {_fmt(retrieval.get('recall_at_5'))}{noise('retrieval', 'recall_at_5')}",
        f"MRR           {_fmt(retrieval.get('mrr'))}{noise('retrieval', 'mrr')}",
        f"nDCG@10       {_fmt(retrieval.get('ndcg_at_10'))}{noise('retrieval', 'ndcg_at_10')}",
        "",
        _RULE,
        "Generation",
        _RULE,
        f"Faithfulness  {_fmt(generation.get('faithfulness'))}"
        f"{noise('generation', 'faithfulness')}",
        f"Correctness   {_fmt(generation.get('correctness'))}{noise('generation', 'correctness')}",
        "",
        _RULE,
        "Safety",
        _RULE,
        f"Refusal Rate  {_fmt(safety.get('refusal_rate'))}{noise('safety', 'refusal_rate')}",
        f"Fabrications  {_fmt(safety.get('fabrications'), digits=0)}",
        "",
        _RULE,
        "Ops",
        _RULE,
        f"Mean Cost     {_fmt_cost(ops.get('mean_cost_usd'))}{noise('ops', 'mean_cost_usd')}",
        f"P95 Latency   {_fmt(ops.get('p95_latency_ms'), suffix=' ms', digits=1)}"
        f"{noise('ops', 'p95_latency_ms')}",
        _RULE,
    ]

    num_errors = scorecard.get("num_errors", 0)
    if num_errors:
        lines.append(f"WARNING: {num_errors} question(s) errored and were excluded above.")
        lines.append(_RULE)

    return "\n".join(lines)
