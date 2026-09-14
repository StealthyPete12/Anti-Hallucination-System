"""Unit tests for judge-validation agreement metrics (eval/agreement.py).

Cohen's kappa expected values below are hand-computed from the standard
formula so a regression in the arithmetic (not just the API) gets caught.
"""

from __future__ import annotations

import pytest

from eval.agreement import (
    AgreementError,
    build_agreement_report,
    build_confusion,
    cohens_kappa,
    percent_agreement,
)


def test_percent_agreement_perfect_match() -> None:
    labels = ["correct", "incorrect", "partial", "correct"]
    assert percent_agreement(labels, labels) == 1.0


def test_percent_agreement_no_match() -> None:
    human = ["correct", "correct"]
    judge = ["incorrect", "incorrect"]
    assert percent_agreement(human, judge) == 0.0


def test_percent_agreement_partial() -> None:
    human = ["correct", "incorrect", "partial", "correct"]
    judge = ["correct", "incorrect", "correct", "incorrect"]
    assert percent_agreement(human, judge) == 0.5


def test_percent_agreement_length_mismatch_raises() -> None:
    with pytest.raises(AgreementError):
        percent_agreement(["correct"], ["correct", "incorrect"])


def test_percent_agreement_empty_raises() -> None:
    with pytest.raises(AgreementError):
        percent_agreement([], [])


def test_cohens_kappa_perfect_agreement_is_one() -> None:
    human = ["correct", "incorrect", "partial", "correct", "incorrect"]
    assert cohens_kappa(human, human) == pytest.approx(1.0)


def test_cohens_kappa_all_same_label_both_raters_is_perfect() -> None:
    # p_observed=1.0, p_expected=1.0 -> defined as 1.0 (trivial perfect agreement).
    human = ["correct"] * 5
    judge = ["correct"] * 5
    assert cohens_kappa(human, judge) == 1.0


def test_cohens_kappa_hand_computed_example() -> None:
    # Built directly from an explicit confusion matrix (counts of
    # (human_label, judge_label) pairs) so the expected p_observed/p_expected
    # below are exact, not narratively-reconstructed from separate lists.
    confusion = {
        ("correct", "correct"): 4,
        ("incorrect", "incorrect"): 3,
        ("partial", "partial"): 1,
        ("correct", "incorrect"): 1,
        ("incorrect", "partial"): 1,
    }
    human: list[str] = []
    judge: list[str] = []
    for (h, j), count in confusion.items():
        human.extend([h] * count)
        judge.extend([j] * count)
    n = len(human)  # 10

    # p_observed = agreeing pairs / n = (4 + 3 + 1) / 10 = 0.8
    assert percent_agreement(human, judge) == pytest.approx(0.8)

    # human marginals: correct=5, incorrect=4, partial=1
    # judge marginals: correct=4, incorrect=4, partial=2
    # p_expected = (5*4 + 4*4 + 1*2) / 10^2 = (20 + 16 + 2) / 100 = 0.38
    p_expected = (5 * 4 + 4 * 4 + 1 * 2) / (n * n)
    expected_kappa = (0.8 - p_expected) / (1 - p_expected)
    assert cohens_kappa(human, judge) == pytest.approx(expected_kappa)


def test_cohens_kappa_length_mismatch_raises() -> None:
    with pytest.raises(AgreementError):
        cohens_kappa(["correct"], ["correct", "incorrect"])


def test_build_confusion_counts_pairs() -> None:
    human = ["correct", "correct", "incorrect"]
    judge = ["correct", "incorrect", "incorrect"]
    confusion = build_confusion(human, judge)
    assert confusion == {
        ("correct", "correct"): 1,
        ("correct", "incorrect"): 1,
        ("incorrect", "incorrect"): 1,
    }


def test_build_agreement_report_needs_rubric_review_below_80_percent() -> None:
    human = ["correct"] * 6 + ["incorrect"] * 4
    judge = ["correct"] * 3 + ["incorrect"] * 7  # only 6/10 truly agree on "correct" set overlap
    report = build_agreement_report(human, judge)
    assert report.sample_size == 10
    if report.percent_agreement < 0.80:
        assert report.needs_rubric_review is True


def test_build_agreement_report_passes_review_at_or_above_80_percent() -> None:
    human = ["correct"] * 8 + ["incorrect"] * 2
    judge = ["correct"] * 8 + ["incorrect"] * 2
    report = build_agreement_report(human, judge)
    assert report.percent_agreement == 1.0
    assert report.needs_rubric_review is False
