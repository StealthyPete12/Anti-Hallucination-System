"""Renders an :class:`eval.compare.ComparisonResult` as GitHub-flavored
markdown suitable for posting directly as a pull request comment.

No I/O here beyond string building - ``eval.compare.main`` writes the
returned string to ``--out``; ``.github/workflows/eval-pr.yml`` posts that
file's contents as the PR comment body.
"""

from __future__ import annotations

from typing import Any

from eval.compare import ComparisonResult, MetricRow, RuleResult

_STATUS_ICON = {
    "improved": "🟢",
    "within_noise_floor": "🟡",
    "regression": "🔴",
    "no_data": "⚪",
}
_STATUS_LABEL = {
    "improved": "Improved",
    "within_noise_floor": "Within Noise Floor",
    "regression": "Regression",
    "no_data": "No data",
}


def _fmt_value(value: float | int | None, metric: str) -> str:
    if value is None:
        return "n/a"
    if metric == "safety.fabrications":
        return str(int(value))
    if metric == "ops.mean_cost_usd":
        return f"${value:.4f}"
    if metric == "ops.p95_latency_ms":
        return f"{value:.1f} ms"
    return f"{value:.3f}"


def _fmt_delta(delta: float | None, metric: str) -> str:
    if delta is None:
        return "n/a"
    sign = "+" if delta >= 0 else ""
    if metric == "ops.mean_cost_usd":
        return f"{sign}${delta:.4f}"
    if metric == "ops.p95_latency_ms":
        return f"{sign}{delta:.1f} ms"
    if metric == "safety.fabrications":
        return f"{sign}{int(delta)}"
    return f"{sign}{delta:.3f}"


def _row_status(row: MetricRow) -> str:
    """A row's status, but a hard/soft gate failure on that exact metric
    always overrides the plain noise-floor comparison with 🚫."""
    return _STATUS_ICON[row.status]


def _metric_table(rows: list[MetricRow], gate_by_metric: dict[str, RuleResult]) -> list[str]:
    lines = ["Metric | Baseline | Candidate | Delta | Status", "---|---|---|---|---"]
    for row in rows:
        status_icon = _row_status(row)
        gate = gate_by_metric.get(row.metric)
        if gate is not None and gate.triggered:
            status_icon = "🚫" if gate.hard else f"{status_icon} ⚠️"
        lines.append(
            f"{row.label} | {_fmt_value(row.baseline_value, row.metric)} | "
            f"{_fmt_value(row.candidate_value, row.metric)} | "
            f"{_fmt_delta(row.delta, row.metric)} | {status_icon}"
        )
    return lines


def render_report(
    result: ComparisonResult, baseline: dict[str, Any], candidate: dict[str, Any]
) -> str:
    gate_by_metric = {r.rule.metric: r for r in result.rule_results}
    sections: dict[str, list[MetricRow]] = {}
    for row in result.rows:
        sections.setdefault(row.section, []).append(row)

    lines: list[str] = ["# Evaluation Results", ""]

    verdict = "🚫 **Gate Failed**" if not result.gate_passed else "✅ **Gate Passed**"
    lines.append(verdict)
    lines.append("")
    lines.append(
        f"Baseline `{baseline.get('git_sha', 'unknown')}` → "
        f"Candidate `{candidate.get('git_sha', 'unknown')}`"
    )
    lines.append("")

    for section in ("Retrieval", "Generation", "Safety", "Operations"):
        if section not in sections:
            continue
        lines.append(f"## {section}")
        lines.append("")
        lines.extend(_metric_table(sections[section], gate_by_metric))
        lines.append("")

    lines.append("## Gate Results")
    lines.append("")

    if result.hard_failures:
        lines.append("### 🚫 Hard Failures (block merge)")
        lines.append("")
        for r in result.hard_failures:
            lines.append(f"- **{r.rule.metric}**: {r.reason}")
        lines.append("")

    if result.warnings:
        lines.append("### ⚠️ Warnings (non-blocking)")
        lines.append("")
        for r in result.warnings:
            lines.append(f"- **{r.rule.metric}**: {r.reason}")
        lines.append("")

    if result.passed_rules:
        lines.append("### ✅ Passed Rules")
        lines.append("")
        for r in result.passed_rules:
            lines.append(f"- **{r.rule.metric}**: {r.reason}")
        lines.append("")

    skipped = [r for r in result.rule_results if r.skipped]
    if skipped:
        lines.append("### ⚪ Skipped (missing data)")
        lines.append("")
        for r in skipped:
            lines.append(f"- **{r.rule.metric}**: {r.reason}")
        lines.append("")

    lines.append("---")
    lines.append(
        "_🟢 Improved · 🟡 Within Noise Floor · 🔴 Regression · 🚫 Gate Failure_"
    )

    return "\n".join(lines)
