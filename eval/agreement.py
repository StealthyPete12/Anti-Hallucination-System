"""Judge validation: percentage agreement and Cohen's kappa between the
LLM judge's correctness verdicts and a human-labeled validation sample.

"If agreement is < 80%, the judging rubric should be improved. Do not
immediately blame the pipeline." - Phase 4 spec, section 12.

Deliberately dependency-free (no ``scikit-learn``/``scipy``): Cohen's kappa
for two raters over a fixed, small label set is a handful of arithmetic
over a confusion matrix, not worth a heavy dependency.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

LABELS: tuple[str, ...] = ("correct", "incorrect", "partial")


class AgreementError(ValueError):
    """Raised when two label sequences can't be compared (length mismatch, unknown label)."""


@dataclass(frozen=True)
class AgreementReport:
    """Human-vs-judge agreement over one labeled sample."""

    sample_size: int
    percent_agreement: float
    cohens_kappa: float
    confusion: dict[tuple[str, str], int]  # (human_label, judge_label) -> count

    @property
    def needs_rubric_review(self) -> bool:
        """< 80% agreement means the rubric (not necessarily the pipeline) should be reviewed."""
        return self.percent_agreement < 0.80


def percent_agreement(human_labels: list[str], judge_labels: list[str]) -> float:
    """Fraction of items where the two raters gave the exact same label."""
    if len(human_labels) != len(judge_labels):
        raise AgreementError(
            f"human_labels ({len(human_labels)}) and judge_labels ({len(judge_labels)}) "
            "must be the same length"
        )
    if not human_labels:
        raise AgreementError("Cannot compute agreement over an empty sample")
    agreements = sum(1 for h, j in zip(human_labels, judge_labels, strict=True) if h == j)
    return agreements / len(human_labels)


def cohens_kappa(human_labels: list[str], judge_labels: list[str]) -> float:
    """Cohen's kappa: agreement corrected for the agreement expected by chance.

    kappa = (p_observed - p_expected) / (1 - p_expected)

    ``p_expected`` is computed from each rater's own marginal label
    frequencies (the standard unweighted Cohen's kappa for two raters).
    Returns ``1.0`` if both raters agree on everything AND every item has
    the same label (p_expected == 1, 0/0 case - perfect, trivial agreement).
    """
    if len(human_labels) != len(judge_labels):
        raise AgreementError(
            f"human_labels ({len(human_labels)}) and judge_labels ({len(judge_labels)}) "
            "must be the same length"
        )
    n = len(human_labels)
    if n == 0:
        raise AgreementError("Cannot compute kappa over an empty sample")

    p_observed = percent_agreement(human_labels, judge_labels)

    human_counts = Counter(human_labels)
    judge_counts = Counter(judge_labels)
    all_labels = set(human_counts) | set(judge_counts)
    p_expected = sum((human_counts[label] / n) * (judge_counts[label] / n) for label in all_labels)

    if p_expected == 1.0:
        return 1.0
    return (p_observed - p_expected) / (1 - p_expected)


def build_confusion(human_labels: list[str], judge_labels: list[str]) -> dict[tuple[str, str], int]:
    return dict(Counter(zip(human_labels, judge_labels, strict=True)))


def build_agreement_report(human_labels: list[str], judge_labels: list[str]) -> AgreementReport:
    return AgreementReport(
        sample_size=len(human_labels),
        percent_agreement=percent_agreement(human_labels, judge_labels),
        cohens_kappa=cohens_kappa(human_labels, judge_labels),
        confusion=build_confusion(human_labels, judge_labels),
    )


def render_report(report: AgreementReport) -> str:
    lines = [
        "# Judge Validation: Human vs. LLM-Judge Agreement",
        "",
        f"Sample size: {report.sample_size}",
        f"Percentage agreement: {report.percent_agreement:.1%}",
        f"Cohen's kappa: {report.cohens_kappa:.3f}",
        "",
    ]
    if report.needs_rubric_review:
        lines.append(
            "⚠️ Agreement is below 80% - review the judging rubric/prompt before concluding "
            "the pipeline itself regressed."
        )
    else:
        lines.append(
            "✅ Agreement is at or above 80% - the judge's correctness rubric is validated."
        )
    lines.append("")
    lines.append("## Confusion matrix (human_label, judge_label) -> count")
    lines.append("")
    for (human, judge), count in sorted(report.confusion.items()):
        lines.append(f"- ({human}, {judge}): {count}")
    return "\n".join(lines)
