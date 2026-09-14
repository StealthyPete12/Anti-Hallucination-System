"""Phase 3 CI quality gate: compares a candidate scorecard (``eval.run``
output) against ``evals/baseline.json`` and decides whether a change is safe
to merge.

    uv run python -m eval.compare \\
        --baseline evals/baseline.json \\
        --candidate evals/results.json \\
        --out comment.md

Exits ``0`` when no hard rule is violated (warnings are still reported, but
never block), and ``1`` when any hard rule fires - that exit code is what
``.github/workflows/eval-pr.yml`` uses to fail the CI check.

Pure comparison logic, no I/O beyond reading the two JSON files and writing
``--out`` - fully unit-testable against hand-built scorecard dicts, without a
live Qdrant instance or LLM provider key.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Metrics where a *smaller* value is the good direction. Everything else
#: (recall, mrr, ndcg, faithfulness, correctness, refusal_rate) is bigger-is-
#: better.
LOWER_IS_BETTER: frozenset[str] = frozenset(
    {"ops.mean_cost_usd", "ops.p95_latency_ms", "safety.fabrications"}
)

#: Every headline metric the report table covers, grouped the way the
#: scorecard (and the Phase 2 README) already groups them.
REPORT_METRICS: tuple[tuple[str, str, str], ...] = (
    ("Retrieval", "Recall@5", "retrieval.recall_at_5"),
    ("Retrieval", "MRR", "retrieval.mrr"),
    ("Retrieval", "nDCG@10", "retrieval.ndcg_at_10"),
    ("Generation", "Faithfulness", "generation.faithfulness"),
    ("Generation", "Correctness", "generation.correctness"),
    ("Safety", "Refusal Rate", "safety.refusal_rate"),
    ("Safety", "Fabrications", "safety.fabrications"),
    ("Operations", "Mean Cost (USD)", "ops.mean_cost_usd"),
    ("Operations", "P95 Latency (ms)", "ops.p95_latency_ms"),
)

#: Fallback noise floor (treated as an absolute-value epsilon, or - for the
#: two ops metrics - a fraction of the baseline value) used only when
#: ``baseline.json`` carries no ``noise_floor`` section of its own (e.g. it
#: was written from a single ``--repeats 1`` run rather than a ``--repeats 5``
#: noise-floor run). Each value is deliberately set to roughly half of the
#: matching GATE rule's threshold below, i.e. the GATE thresholds already
#: encode "~2x this noise floor" - see the README's "Noise floor" section for
#: the full rationale.
FALLBACK_NOISE_FLOOR: dict[str, float] = {
    "retrieval.recall_at_5": 0.015,
    "retrieval.mrr": 0.02,
    "retrieval.ndcg_at_10": 0.02,
    "generation.faithfulness": 0.025,
    "generation.correctness": 0.03,
    "safety.refusal_rate": 0.05,
    "safety.fabrications": 0.0,
    "ops.mean_cost_usd": 0.125,  # fraction of baseline value (12.5%)
    "ops.p95_latency_ms": 0.15,  # fraction of baseline value (15%)
}


def get_metric(scorecard: dict[str, Any], dotted: str) -> float | int | None:
    """Read a ``section.key`` dotted path out of a scorecard dict."""
    section, key = dotted.split(".", 1)
    return scorecard.get(section, {}).get(key)


def get_noise_floor(metric: str, baseline: dict[str, Any]) -> float:
    """The run-to-run standard deviation to treat as noise for ``metric``.

    Prefers the baseline's own measured ``noise_floor`` (written by
    ``eval.scorecard.compute_noise_floor`` when the baseline was captured
    with ``--repeats > 1``) over the hardcoded :data:`FALLBACK_NOISE_FLOOR`
    table, since a real measurement is always better than an estimate.
    """
    measured = baseline.get("noise_floor", {}).get(metric)
    if measured is not None and measured.get("stdev") is not None:
        return float(measured["stdev"])

    fallback = FALLBACK_NOISE_FLOOR.get(metric, 0.0)
    if metric in {"ops.mean_cost_usd", "ops.p95_latency_ms"}:
        baseline_value = get_metric(baseline, metric)
        if baseline_value:
            return fallback * float(baseline_value)
        return 0.0
    return fallback


@dataclass(frozen=True)
class Rule:
    """One gate rule, evaluated for a single dotted metric path.

    Exactly one of ``max_drop``, ``max_absolute``, ``max_increase_pct``
    should normally be set (multiple may be set; a rule triggers if *any*
    configured bound is violated). ``hard=True`` rules fail CI;
    ``hard=False`` rules are reported as warnings only.
    """

    metric: str
    max_drop: float | None = None
    max_absolute: float | None = None
    max_increase_pct: float | None = None
    hard: bool = True

    def __post_init__(self) -> None:
        if self.max_drop is None and self.max_absolute is None and self.max_increase_pct is None:
            raise ValueError(f"Rule({self.metric!r}) has no bound configured")


#: The active gate. See the README's "CI Quality Gate" section for the
#: rationale behind each threshold.
GATE: list[Rule] = [
    Rule("safety.fabrications", max_absolute=0, hard=True),
    Rule("generation.faithfulness", max_drop=0.05, hard=True),
    Rule("retrieval.recall_at_5", max_drop=0.03, hard=True),
    Rule("safety.refusal_rate", max_drop=0.10, hard=True),
    Rule("ops.mean_cost_usd", max_increase_pct=25, hard=False),
    Rule("ops.p95_latency_ms", max_increase_pct=30, hard=False),
]


@dataclass(frozen=True)
class RuleResult:
    """The outcome of evaluating one :class:`Rule` against a comparison."""

    rule: Rule
    baseline_value: float | int | None
    candidate_value: float | int | None
    delta: float | None
    triggered: bool
    reason: str

    @property
    def hard(self) -> bool:
        return self.rule.hard

    @property
    def skipped(self) -> bool:
        return self.baseline_value is None or self.candidate_value is None


@dataclass(frozen=True)
class MetricRow:
    """One report-table row: a headline metric's baseline/candidate/delta
    plus a noise-floor-aware status, independent of whether a GATE rule
    happens to cover that metric."""

    section: str
    label: str
    metric: str
    baseline_value: float | int | None
    candidate_value: float | int | None
    delta: float | None
    status: str  # "improved" | "within_noise_floor" | "regression" | "no_data"


@dataclass(frozen=True)
class ComparisonResult:
    rows: list[MetricRow]
    rule_results: list[RuleResult]

    @property
    def hard_failures(self) -> list[RuleResult]:
        return [r for r in self.rule_results if r.triggered and r.hard]

    @property
    def warnings(self) -> list[RuleResult]:
        return [r for r in self.rule_results if r.triggered and not r.hard]

    @property
    def passed_rules(self) -> list[RuleResult]:
        return [r for r in self.rule_results if not r.triggered and not r.skipped]

    @property
    def gate_passed(self) -> bool:
        return len(self.hard_failures) == 0


def evaluate_rule(
    rule: Rule, baseline_value: float | int | None, candidate_value: float | int | None
) -> RuleResult:
    """Check one rule's bounds. A rule with a missing baseline or candidate
    value never triggers (there's nothing to compare) but is reported as
    skipped so a silently-absent metric doesn't read as a silent pass."""
    if baseline_value is None or candidate_value is None:
        return RuleResult(
            rule, baseline_value, candidate_value, None, False, "skipped: missing value"
        )

    delta = float(candidate_value) - float(baseline_value)
    reasons: list[str] = []

    if rule.max_drop is not None and delta < -rule.max_drop:
        reasons.append(f"dropped {-delta:.4f} (max_drop={rule.max_drop})")

    if rule.max_absolute is not None and float(candidate_value) > rule.max_absolute:
        reasons.append(f"value {candidate_value} exceeds max_absolute={rule.max_absolute}")

    if rule.max_increase_pct is not None and baseline_value:
        pct = (delta / abs(float(baseline_value))) * 100
        if pct > rule.max_increase_pct:
            reasons.append(f"increased {pct:.1f}% (max_increase_pct={rule.max_increase_pct}%)")

    triggered = bool(reasons)
    reason = "; ".join(reasons) if reasons else "within bounds"
    return RuleResult(rule, baseline_value, candidate_value, delta, triggered, reason)


def _status_for(metric: str, delta: float, noise_floor: float) -> str:
    direction = -1.0 if metric in LOWER_IS_BETTER else 1.0
    signed = delta * direction  # positive = change in the good direction
    if abs(delta) <= noise_floor:
        return "within_noise_floor"
    return "improved" if signed > 0 else "regression"


def build_metric_rows(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[MetricRow]:
    rows: list[MetricRow] = []
    for section, label, metric in REPORT_METRICS:
        baseline_value = get_metric(baseline, metric)
        candidate_value = get_metric(candidate, metric)
        if baseline_value is None or candidate_value is None:
            rows.append(
                MetricRow(section, label, metric, baseline_value, candidate_value, None, "no_data")
            )
            continue
        delta = float(candidate_value) - float(baseline_value)
        noise_floor = get_noise_floor(metric, baseline)
        status = _status_for(metric, delta, noise_floor)
        rows.append(
            MetricRow(section, label, metric, baseline_value, candidate_value, delta, status)
        )
    return rows


def compare_scorecards(
    baseline: dict[str, Any], candidate: dict[str, Any], *, gate: list[Rule] | None = None
) -> ComparisonResult:
    """Compare a baseline and candidate scorecard and evaluate every gate
    rule. This is the single entry point both the CLI and tests use."""
    rules = gate if gate is not None else GATE
    rows = build_metric_rows(baseline, candidate)
    rule_results = [
        evaluate_rule(rule, get_metric(baseline, rule.metric), get_metric(candidate, rule.metric))
        for rule in rules
    ]
    return ComparisonResult(rows=rows, rule_results=rule_results)


def load_scorecard(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a candidate scorecard against evals/baseline.json and enforce the "
        "Phase 3 CI quality gate."
    )
    parser.add_argument("--baseline", required=True, help="Path to the baseline scorecard JSON.")
    parser.add_argument("--candidate", required=True, help="Path to the candidate scorecard JSON.")
    parser.add_argument("--out", default=None, help="Write the markdown PR report here.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    baseline = load_scorecard(args.baseline)
    candidate = load_scorecard(args.candidate)
    result = compare_scorecards(baseline, candidate)

    if args.out:
        from eval.report import render_report

        Path(args.out).write_text(render_report(result, baseline, candidate), encoding="utf-8")

    for rule_result in result.hard_failures:
        print(f"GATE FAILURE: {rule_result.rule.metric}: {rule_result.reason}", file=sys.stderr)
    for rule_result in result.warnings:
        print(f"WARNING: {rule_result.rule.metric}: {rule_result.reason}", file=sys.stderr)

    if not result.gate_passed:
        print(
            f"\n{len(result.hard_failures)} hard gate failure(s). See above for details.",
            file=sys.stderr,
        )
        return 1

    print("Gate passed: no hard regressions detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
