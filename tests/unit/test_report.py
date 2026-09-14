"""Unit tests for eval.report: markdown rendering of a ComparisonResult."""

from __future__ import annotations

from typing import Any

from eval.compare import compare_scorecards
from eval.report import render_report


def make_scorecard(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "git_sha": "abc123",
        "retrieval": {"recall_at_5": 0.82, "mrr": 0.68, "ndcg_at_10": 0.74},
        "generation": {"faithfulness": 0.89, "correctness": 0.79},
        "safety": {"refusal_rate": 0.93, "fabrications": 0},
        "ops": {"mean_cost_usd": 0.0019, "p95_latency_ms": 1740.0},
    }
    for dotted, value in overrides.items():
        section, key = dotted.split(".", 1)
        base.setdefault(section, {})[key] = value
    return base


def test_report_includes_all_sections() -> None:
    baseline = make_scorecard()
    candidate = make_scorecard()
    result = compare_scorecards(baseline, candidate)
    report = render_report(result, baseline, candidate)
    for heading in ("## Retrieval", "## Generation", "## Safety", "## Operations", "## Gate Results"):
        assert heading in report


def test_report_shows_gate_passed_when_no_hard_failures() -> None:
    baseline = make_scorecard()
    candidate = make_scorecard()
    result = compare_scorecards(baseline, candidate)
    report = render_report(result, baseline, candidate)
    assert "Gate Passed" in report
    assert "Gate Failed" not in report


def test_report_shows_gate_failed_and_lists_hard_failure() -> None:
    baseline = make_scorecard()
    candidate = make_scorecard(**{"generation.faithfulness": 0.60})
    result = compare_scorecards(baseline, candidate)
    report = render_report(result, baseline, candidate)
    assert "Gate Failed" in report
    assert "🚫 Hard Failures" in report
    assert "generation.faithfulness" in report


def test_report_lists_warnings_separately_from_hard_failures() -> None:
    baseline = make_scorecard()
    candidate = make_scorecard(**{"ops.mean_cost_usd": 0.0040})
    result = compare_scorecards(baseline, candidate)
    report = render_report(result, baseline, candidate)
    assert "⚠️ Warnings" in report
    assert "Gate Passed" in report


def test_report_includes_git_shas() -> None:
    baseline = make_scorecard()
    candidate = make_scorecard()
    candidate["git_sha"] = "def456"
    result = compare_scorecards(baseline, candidate)
    report = render_report(result, baseline, candidate)
    assert "abc123" in report
    assert "def456" in report


def test_report_marks_regression_metric_with_gate_failure_icon() -> None:
    baseline = make_scorecard()
    candidate = make_scorecard(**{"retrieval.recall_at_5": 0.50})
    result = compare_scorecards(baseline, candidate)
    report = render_report(result, baseline, candidate)
    recall_line = next(line for line in report.splitlines() if line.startswith("Recall@5"))
    assert "🚫" in recall_line
