"""Unit tests for eval.compare: the Phase 3 CI quality gate engine.

Everything here works against hand-built scorecard dicts (the same shape
eval.scorecard.build_scorecard produces) - no Qdrant, no LLM provider, no
eval.run involved.
"""

from __future__ import annotations

from typing import Any

import pytest

from eval.compare import GATE, Rule, compare_scorecards, evaluate_rule, get_metric


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


class TestGetMetric:
    def test_reads_nested_value(self) -> None:
        sc = make_scorecard()
        assert get_metric(sc, "retrieval.recall_at_5") == 0.82

    def test_missing_section_returns_none(self) -> None:
        assert get_metric({}, "retrieval.recall_at_5") is None


class TestEvaluateRule:
    def test_max_drop_not_triggered_within_bound(self) -> None:
        rule = Rule("generation.faithfulness", max_drop=0.05)
        result = evaluate_rule(rule, 0.89, 0.86)
        assert not result.triggered

    def test_max_drop_triggered_beyond_bound(self) -> None:
        rule = Rule("generation.faithfulness", max_drop=0.05)
        result = evaluate_rule(rule, 0.89, 0.80)
        assert result.triggered
        assert "dropped" in result.reason

    def test_max_drop_improvement_never_triggers(self) -> None:
        rule = Rule("generation.faithfulness", max_drop=0.05)
        result = evaluate_rule(rule, 0.80, 0.95)
        assert not result.triggered

    def test_max_absolute_triggered(self) -> None:
        rule = Rule("safety.fabrications", max_absolute=0)
        result = evaluate_rule(rule, 0, 1)
        assert result.triggered

    def test_max_absolute_not_triggered_at_bound(self) -> None:
        rule = Rule("safety.fabrications", max_absolute=0)
        result = evaluate_rule(rule, 0, 0)
        assert not result.triggered

    def test_max_increase_pct_triggered(self) -> None:
        rule = Rule("ops.mean_cost_usd", max_increase_pct=25, hard=False)
        result = evaluate_rule(rule, 0.0020, 0.0030)  # +50%
        assert result.triggered

    def test_max_increase_pct_not_triggered_within_bound(self) -> None:
        rule = Rule("ops.mean_cost_usd", max_increase_pct=25, hard=False)
        result = evaluate_rule(rule, 0.0020, 0.0022)  # +10%
        assert not result.triggered

    def test_missing_value_skips_without_triggering(self) -> None:
        rule = Rule("generation.faithfulness", max_drop=0.05)
        result = evaluate_rule(rule, None, 0.80)
        assert not result.triggered
        assert result.skipped

    def test_rule_requires_a_bound(self) -> None:
        with pytest.raises(ValueError):
            Rule("generation.faithfulness")


class TestCompareScorecards:
    def test_identical_scorecards_pass_gate(self) -> None:
        baseline = make_scorecard()
        candidate = make_scorecard()
        result = compare_scorecards(baseline, candidate)
        assert result.gate_passed
        assert not result.hard_failures

    def test_faithfulness_regression_is_a_hard_failure(self) -> None:
        baseline = make_scorecard()
        candidate = make_scorecard(**{"generation.faithfulness": 0.70})
        result = compare_scorecards(baseline, candidate)
        assert not result.gate_passed
        assert any(r.rule.metric == "generation.faithfulness" for r in result.hard_failures)

    def test_recall_regression_is_a_hard_failure(self) -> None:
        baseline = make_scorecard()
        candidate = make_scorecard(**{"retrieval.recall_at_5": 0.70})
        result = compare_scorecards(baseline, candidate)
        assert not result.gate_passed
        assert any(r.rule.metric == "retrieval.recall_at_5" for r in result.hard_failures)

    def test_refusal_regression_is_a_hard_failure(self) -> None:
        baseline = make_scorecard()
        candidate = make_scorecard(**{"safety.refusal_rate": 0.75})
        result = compare_scorecards(baseline, candidate)
        assert not result.gate_passed
        assert any(r.rule.metric == "safety.refusal_rate" for r in result.hard_failures)

    def test_any_fabrication_is_a_hard_failure(self) -> None:
        baseline = make_scorecard()
        candidate = make_scorecard(**{"safety.fabrications": 1})
        result = compare_scorecards(baseline, candidate)
        assert not result.gate_passed
        assert any(r.rule.metric == "safety.fabrications" for r in result.hard_failures)

    def test_cost_regression_is_a_warning_not_a_failure(self) -> None:
        baseline = make_scorecard()
        candidate = make_scorecard(**{"ops.mean_cost_usd": 0.0030})  # +58%
        result = compare_scorecards(baseline, candidate)
        assert result.gate_passed
        assert any(r.rule.metric == "ops.mean_cost_usd" for r in result.warnings)

    def test_latency_regression_is_a_warning_not_a_failure(self) -> None:
        baseline = make_scorecard()
        candidate = make_scorecard(**{"ops.p95_latency_ms": 3000.0})
        result = compare_scorecards(baseline, candidate)
        assert result.gate_passed
        assert any(r.rule.metric == "ops.p95_latency_ms" for r in result.warnings)

    def test_improvement_is_reported_as_improved_row(self) -> None:
        baseline = make_scorecard()
        candidate = make_scorecard(**{"generation.faithfulness": 0.99})
        result = compare_scorecards(baseline, candidate)
        row = next(r for r in result.rows if r.metric == "generation.faithfulness")
        assert row.status == "improved"

    def test_tiny_change_is_within_noise_floor(self) -> None:
        baseline = make_scorecard()
        candidate = make_scorecard(**{"generation.correctness": 0.795})
        result = compare_scorecards(baseline, candidate)
        row = next(r for r in result.rows if r.metric == "generation.correctness")
        assert row.status == "within_noise_floor"

    def test_missing_candidate_metric_reports_no_data(self) -> None:
        baseline = make_scorecard()
        candidate = make_scorecard()
        del candidate["generation"]["correctness"]
        result = compare_scorecards(baseline, candidate)
        row = next(r for r in result.rows if r.metric == "generation.correctness")
        assert row.status == "no_data"

    def test_uses_measured_noise_floor_when_present_on_baseline(self) -> None:
        baseline = make_scorecard()
        baseline["noise_floor"] = {
            "generation.faithfulness": {"mean": 0.89, "stdev": 0.2, "n": 5}
        }
        candidate = make_scorecard(**{"generation.faithfulness": 0.75})
        result = compare_scorecards(baseline, candidate)
        row = next(r for r in result.rows if r.metric == "generation.faithfulness")
        # A 0.14 drop is within a measured stdev of 0.2, so it reads as noise
        # even though the fixed GATE rule (max_drop=0.05) still hard-fails it.
        assert row.status == "within_noise_floor"
        assert not result.gate_passed

    def test_default_gate_is_the_documented_gate(self) -> None:
        assert compare_scorecards(make_scorecard(), make_scorecard()).rule_results
        assert len(GATE) == 6
