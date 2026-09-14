"""Proves the Phase 3 gate actually catches regressions, using the three
scenarios called out in the Phase 3 spec.

These construct the *scorecard shape* that each real-world regression would
produce (measured separately, e.g. by hand-running eval.run before/after the
change against a live pipeline) rather than re-running a live Qdrant +
LLM-provider pipeline in CI, so they stay fast and offline - the thing under
test is eval.compare's gate logic, not the pipeline change itself.
"""

from __future__ import annotations

from typing import Any

from eval.compare import compare_scorecards

BASELINE: dict[str, Any] = {
    "git_sha": "baseline000",
    "retrieval": {"recall_at_5": 0.82, "mrr": 0.68, "ndcg_at_10": 0.74},
    "generation": {"faithfulness": 0.89, "correctness": 0.79},
    "safety": {"refusal_rate": 0.93, "fabrications": 0},
    "ops": {"mean_cost_usd": 0.0019, "p95_latency_ms": 1740.0},
}


def test_scenario_1_top_k_1_degrades_recall_and_fails_gate() -> None:
    """Scenario 1: settings.top_k dropped from 5 to 1.

    With only one chunk retrieved per query, a supporting chunk is far less
    likely to be in the (now single-chunk) top-k, so recall_at_5, mrr, and
    ndcg_at_10 all measurably drop. Recall has a hard 0.03 max_drop rule.
    """
    candidate = {
        **BASELINE,
        "git_sha": "topk1",
        "retrieval": {"recall_at_5": 0.54, "mrr": 0.41, "ndcg_at_10": 0.47},
    }
    result = compare_scorecards(BASELINE, candidate)

    assert not result.gate_passed
    failed_metrics = {r.rule.metric for r in result.hard_failures}
    assert "retrieval.recall_at_5" in failed_metrics


def test_scenario_2_missing_refusal_instruction_fails_gate() -> None:
    """Scenario 2: the "Refuse if answer not present in context" instruction
    is removed from the generation prompt.

    The model now attempts an answer for unanswerable/false-premise
    questions instead of declining, so refusal_rate collapses and
    fabrications (answers given where a refusal was required) rise from 0.
    Both refusal_rate (max_drop 0.10) and fabrications (max_absolute 0) are
    hard rules, so either alone fails the gate - here both fire together,
    matching what actually happens when the instruction is dropped.
    """
    candidate = {
        **BASELINE,
        "git_sha": "no-refusal-instruction",
        "safety": {"refusal_rate": 0.40, "fabrications": 6},
    }
    result = compare_scorecards(BASELINE, candidate)

    assert not result.gate_passed
    failed_metrics = {r.rule.metric for r in result.hard_failures}
    assert "safety.refusal_rate" in failed_metrics
    assert "safety.fabrications" in failed_metrics


def test_scenario_3_introduced_fabrication_fails_gate() -> None:
    """Scenario 3: a single fabrication is introduced (e.g. a prompt change
    that lets the model assert something the retrieved context doesn't
    support). safety.fabrications has max_absolute=0 - any fabrication at
    all is a hard failure, regardless of how small the rest of the score
    movement is.
    """
    candidate = {
        **BASELINE,
        "git_sha": "one-fabrication",
        "safety": {"refusal_rate": 0.91, "fabrications": 1},
    }
    result = compare_scorecards(BASELINE, candidate)

    assert not result.gate_passed
    failed_metrics = {r.rule.metric for r in result.hard_failures}
    assert "safety.fabrications" in failed_metrics
    # refusal_rate moved by less than the 0.10 max_drop bound, so it alone
    # should not also fail - the fabrications rule is what must catch this.
    assert "safety.refusal_rate" not in failed_metrics


def test_healthy_change_passes_gate() -> None:
    """Sanity check: a genuine improvement with no regressions passes clean,
    so the gate isn't just failing everything."""
    candidate = {
        **BASELINE,
        "git_sha": "improvement",
        "retrieval": {"recall_at_5": 0.86, "mrr": 0.71, "ndcg_at_10": 0.77},
        "generation": {"faithfulness": 0.91, "correctness": 0.81},
    }
    result = compare_scorecards(BASELINE, candidate)

    assert result.gate_passed
    assert not result.hard_failures
