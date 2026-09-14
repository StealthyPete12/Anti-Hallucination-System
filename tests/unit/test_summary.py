"""Unit tests for eval/summary.py: the human-readable scorecard summary."""

from __future__ import annotations

from eval.summary import render_summary


def make_scorecard(**overrides: object) -> dict[str, object]:
    defaults: dict[str, object] = {
        "retrieval": {"recall_at_5": 0.82, "mrr": 0.68, "ndcg_at_10": 0.74},
        "generation": {"faithfulness": 0.89, "correctness": 0.79},
        "safety": {"refusal_rate": 0.93, "fabrications": 1},
        "ops": {"mean_cost_usd": 0.0019, "p95_latency_ms": 1740.0},
        "num_errors": 0,
    }
    defaults.update(overrides)
    return defaults


def test_render_summary_contains_all_sections() -> None:
    text = render_summary(make_scorecard())
    assert "Retrieval" in text
    assert "Generation" in text
    assert "Safety" in text
    assert "Ops" in text


def test_render_summary_contains_metric_values() -> None:
    text = render_summary(make_scorecard())
    assert "0.820" in text
    assert "0.680" in text
    assert "0.740" in text
    assert "0.890" in text
    assert "0.790" in text
    assert "0.930" in text
    assert "1" in text  # fabrications
    assert "$0.0019" in text
    assert "1740.0 ms" in text


def test_render_summary_handles_none_values() -> None:
    sc = make_scorecard(retrieval={"recall_at_5": None, "mrr": None, "ndcg_at_10": None})
    text = render_summary(sc)
    assert "n/a" in text


def test_render_summary_warns_on_errors() -> None:
    text = render_summary(make_scorecard(num_errors=3))
    assert "WARNING" in text
    assert "3 question(s) errored" in text


def test_render_summary_no_warning_when_no_errors() -> None:
    text = render_summary(make_scorecard(num_errors=0))
    assert "WARNING" not in text


def test_render_summary_includes_noise_floor_when_provided() -> None:
    noise_floor = {
        "retrieval.recall_at_5": {"mean": 0.82, "stdev": 0.01, "n": 5},
    }
    text = render_summary(make_scorecard(), noise_floor=noise_floor)
    assert "mean=0.820" in text
    assert "stdev=0.010" in text
    assert "n=5" in text


def test_render_summary_reads_noise_floor_from_scorecard_if_not_passed() -> None:
    sc = make_scorecard()
    sc["noise_floor"] = {"retrieval.recall_at_5": {"mean": 0.5, "stdev": 0.02, "n": 3}}
    text = render_summary(sc)
    assert "mean=0.500" in text
