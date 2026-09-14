"""Unit tests for eval/scorecard.py: aggregation and noise-floor computation.

Built entirely from hand-constructed QuestionResult lists / scorecard dicts,
so expected aggregates are verified against manually computed values, not
just re-run through the implementation.
"""

from __future__ import annotations

import pytest

from eval.results import QuestionResult
from eval.scorecard import build_scorecard, compute_noise_floor


def make_result(**overrides: object) -> QuestionResult:
    defaults: dict[str, object] = {
        "id": "q001",
        "type": "single_hop",
        "question": "q",
        "generated_answer": "a",
        "reference_answer": "r",
        "retrieved_chunk_ids": ["a#0"],
        "retrieval_metrics": {"recall_at_5": 1.0, "mrr": 1.0, "ndcg_at_10": 1.0},
        "faithfulness": {"score": 1.0, "unsupported_count": 0, "claims": []},
        "correctness": {"verdict": "correct", "reason": "", "score": 1.0},
        "refusal": None,
        "latency_ms": {"embed": 10.0, "retrieve": 5.0, "generate": 100.0, "total": 115.0},
        "cost_usd": 0.001,
        "trace_id": "t1",
        "error": None,
    }
    defaults.update(overrides)
    return QuestionResult(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# build_scorecard
# ---------------------------------------------------------------------------


def test_build_scorecard_basic_shape() -> None:
    results = [make_result()]
    sc = build_scorecard(
        results, git_sha="abc123", timestamp="2026-01-01T00:00:00Z", config={"top_k": 5}
    )

    assert sc["git_sha"] == "abc123"
    assert sc["timestamp"] == "2026-01-01T00:00:00Z"
    assert sc["config"] == {"top_k": 5}
    assert set(sc["retrieval"]) == {"recall_at_5", "mrr", "ndcg_at_10"}
    assert set(sc["generation"]) == {"faithfulness", "correctness"}
    assert set(sc["safety"]) == {"refusal_rate", "fabrications"}
    assert set(sc["ops"]) == {"mean_cost_usd", "p95_latency_ms"}
    assert len(sc["per_question"]) == 1


def test_build_scorecard_averages_retrieval_metrics() -> None:
    results = [
        make_result(retrieval_metrics={"recall_at_5": 1.0, "mrr": 1.0, "ndcg_at_10": 1.0}),
        make_result(retrieval_metrics={"recall_at_5": 0.0, "mrr": 0.5, "ndcg_at_10": 0.5}),
    ]
    sc = build_scorecard(results, git_sha="x", timestamp="t", config={})
    assert sc["retrieval"]["recall_at_5"] == pytest.approx(0.5)
    assert sc["retrieval"]["mrr"] == pytest.approx(0.75)
    assert sc["retrieval"]["ndcg_at_10"] == pytest.approx(0.75)


def test_build_scorecard_averages_generation_metrics() -> None:
    results = [
        make_result(
            faithfulness={"score": 1.0, "unsupported_count": 0, "claims": []},
            correctness={"verdict": "correct", "reason": "", "score": 1.0},
        ),
        make_result(
            faithfulness={"score": 0.5, "unsupported_count": 1, "claims": []},
            correctness={"verdict": "incorrect", "reason": "", "score": 0.0},
        ),
    ]
    sc = build_scorecard(results, git_sha="x", timestamp="t", config={})
    assert sc["generation"]["faithfulness"] == pytest.approx(0.75)
    assert sc["generation"]["correctness"] == pytest.approx(0.5)


def test_build_scorecard_refusal_rate_and_fabrications() -> None:
    results = [
        make_result(id="q1", type="unanswerable", refusal={"refused": True, "reason": ""}),
        make_result(id="q2", type="unanswerable", refusal={"refused": False, "reason": ""}),
        make_result(id="q3", type="single_hop", refusal=None),
    ]
    sc = build_scorecard(results, git_sha="x", timestamp="t", config={})
    assert sc["safety"]["refusal_rate"] == pytest.approx(0.5)
    assert sc["safety"]["fabrications"] == 1


def test_build_scorecard_refusal_rate_none_when_no_refusal_questions() -> None:
    results = [make_result(refusal=None)]
    sc = build_scorecard(results, git_sha="x", timestamp="t", config={})
    assert sc["safety"]["refusal_rate"] is None
    assert sc["safety"]["fabrications"] == 0


def test_build_scorecard_ops_metrics() -> None:
    results = [
        make_result(cost_usd=0.001, latency_ms={"total": 100.0}),
        make_result(cost_usd=0.003, latency_ms={"total": 300.0}),
    ]
    sc = build_scorecard(results, git_sha="x", timestamp="t", config={})
    assert sc["ops"]["mean_cost_usd"] == pytest.approx(0.002)
    assert sc["ops"]["p95_latency_ms"] == pytest.approx(290.0)  # linear interpolation, 2 points


def test_build_scorecard_excludes_errored_questions_from_aggregates() -> None:
    results = [
        make_result(id="q1", retrieval_metrics={"recall_at_5": 1.0, "mrr": 1.0, "ndcg_at_10": 1.0}),
        make_result(
            id="q2",
            error="pipeline query failed: timeout",
            retrieval_metrics={"recall_at_5": None, "mrr": None, "ndcg_at_10": None},
        ),
    ]
    sc = build_scorecard(results, git_sha="x", timestamp="t", config={})
    assert sc["retrieval"]["recall_at_5"] == pytest.approx(1.0)  # only q1 counted
    assert sc["num_questions"] == 2
    assert sc["num_errors"] == 1
    assert len(sc["per_question"]) == 2  # still present for debugging


def test_build_scorecard_none_retrieval_values_excluded_from_mean() -> None:
    # e.g. unanswerable questions have no supporting_chunk_ids -> recall is None
    results = [
        make_result(id="q1", retrieval_metrics={"recall_at_5": 1.0, "mrr": 1.0, "ndcg_at_10": 1.0}),
        make_result(
            id="q2",
            type="unanswerable",
            retrieval_metrics={"recall_at_5": None, "mrr": None, "ndcg_at_10": None},
        ),
    ]
    sc = build_scorecard(results, git_sha="x", timestamp="t", config={})
    assert sc["retrieval"]["recall_at_5"] == pytest.approx(1.0)


def test_build_scorecard_empty_results() -> None:
    sc = build_scorecard([], git_sha="x", timestamp="t", config={})
    assert sc["retrieval"]["recall_at_5"] is None
    assert sc["generation"]["faithfulness"] is None
    assert sc["safety"]["refusal_rate"] is None
    assert sc["ops"]["mean_cost_usd"] is None
    assert sc["num_questions"] == 0


# ---------------------------------------------------------------------------
# compute_noise_floor
# ---------------------------------------------------------------------------


def _scorecard(recall: float, faithfulness: float, cost: float) -> dict[str, object]:
    return {
        "retrieval": {"recall_at_5": recall, "mrr": 0.5, "ndcg_at_10": 0.5},
        "generation": {"faithfulness": faithfulness, "correctness": 0.5},
        "safety": {"refusal_rate": 0.9, "fabrications": 1},
        "ops": {"mean_cost_usd": cost, "p95_latency_ms": 1000.0},
    }


def test_compute_noise_floor_mean_and_stdev() -> None:
    scorecards = [_scorecard(0.8, 0.9, 0.001), _scorecard(0.9, 0.8, 0.002)]
    nf = compute_noise_floor(scorecards)

    assert nf["retrieval.recall_at_5"]["mean"] == pytest.approx(0.85)
    assert nf["retrieval.recall_at_5"]["stdev"] == pytest.approx(0.0707106781, rel=1e-6)
    assert nf["retrieval.recall_at_5"]["n"] == 2


def test_compute_noise_floor_single_run_has_zero_stdev() -> None:
    scorecards = [_scorecard(0.8, 0.9, 0.001)]
    nf = compute_noise_floor(scorecards)
    assert nf["retrieval.recall_at_5"]["mean"] == pytest.approx(0.8)
    assert nf["retrieval.recall_at_5"]["stdev"] == 0.0
    assert nf["retrieval.recall_at_5"]["n"] == 1


def test_compute_noise_floor_no_runs_is_none() -> None:
    nf = compute_noise_floor([])
    assert nf["retrieval.recall_at_5"]["mean"] is None
    assert nf["retrieval.recall_at_5"]["n"] == 0


def test_compute_noise_floor_covers_all_headline_metrics() -> None:
    scorecards = [_scorecard(0.8, 0.9, 0.001), _scorecard(0.9, 0.8, 0.002)]
    nf = compute_noise_floor(scorecards)
    assert set(nf) == {
        "retrieval.recall_at_5",
        "retrieval.mrr",
        "retrieval.ndcg_at_10",
        "generation.faithfulness",
        "generation.correctness",
        "safety.refusal_rate",
        "ops.mean_cost_usd",
        "ops.p95_latency_ms",
    }
