"""Unit tests for the Phase 4 experiment framework's pure logic (eval/experiment.py).

``run_experiment`` itself (which builds a real pipeline, indexes a corpus,
and needs model weights) is exercised in
tests/integration/test_experiment_e2e.py; these tests cover the
LLM-free, pipeline-free pieces: settings-override validation,
retrieval-only scoring, and baseline-delta comparison.
"""

from __future__ import annotations

import pytest

from eval.dataset import GoldenQuestion
from eval.experiment import build_experiment_settings, compare_to_baseline, run_retrieval_only
from rag.models import RetrievedChunk


def make_question(qid: str, supporting: list[str]) -> GoldenQuestion:
    return GoldenQuestion(
        id=qid,
        type="single_hop",
        question=f"question {qid}?",
        reference_answer="ref",
        supporting_chunk_ids=supporting,
        tags=[],
    )


def make_chunk(chunk_id: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("#")[0],
        text="text",
        score=1.0,
        source_path="x.txt",
        heading="",
        chunk_index=0,
    )


class FakeRetrievalPipeline:
    """Satisfies only RagPipeline.retrieve - run_retrieval_only never calls .query()."""

    def __init__(self, results_by_question: dict[str, list[RetrievedChunk]]) -> None:
        self._results = results_by_question

    def retrieve(self, question: str) -> tuple[list[RetrievedChunk], dict[str, float]]:
        return self._results.get(question, []), {"embed": 5.0, "retrieve": 3.0, "rerank": 0.0}


def test_build_experiment_settings_applies_overrides() -> None:
    settings = build_experiment_settings({"top_k": 10, "retrieval_mode": "hybrid"})
    assert settings.top_k == 10
    assert settings.retrieval_mode == "hybrid"


def test_build_experiment_settings_rejects_invalid_retrieval_mode() -> None:
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError, not our type
        build_experiment_settings({"retrieval_mode": "not-a-real-mode"})


def test_build_experiment_settings_preserves_unrelated_defaults() -> None:
    settings = build_experiment_settings({"top_k": 3})
    assert settings.chunk_size_tokens == 512  # untouched default, not clobbered


def test_run_retrieval_only_computes_recall_mrr_ndcg() -> None:
    question = make_question("q1", supporting=["doc#0"])
    pipeline = FakeRetrievalPipeline({"question q1?": [make_chunk("doc#0"), make_chunk("doc#1")]})
    result = run_retrieval_only(pipeline, [question])  # type: ignore[arg-type]
    assert result["retrieval"]["recall_at_5"] == 1.0
    assert result["retrieval"]["mrr"] == 1.0
    assert result["num_questions"] == 1


def test_run_retrieval_only_excludes_unanswerable_from_average() -> None:
    answerable = make_question("q1", supporting=["doc#0"])
    unanswerable = make_question("q2", supporting=[])  # no supporting chunks
    pipeline = FakeRetrievalPipeline(
        {
            "question q1?": [make_chunk("doc#0")],
            "question q2?": [make_chunk("doc#9")],
        }
    )
    result = run_retrieval_only(pipeline, [answerable, unanswerable])  # type: ignore[arg-type]
    # Only the answerable question contributes to the recall average.
    assert result["retrieval"]["recall_at_5"] == 1.0


def test_run_retrieval_only_records_per_question_detail() -> None:
    question = make_question("q1", supporting=["doc#0"])
    pipeline = FakeRetrievalPipeline({"question q1?": [make_chunk("doc#5")]})
    result = run_retrieval_only(pipeline, [question])  # type: ignore[arg-type]
    assert result["per_question"][0]["id"] == "q1"
    assert result["per_question"][0]["recall_at_5"] == 0.0


def test_compare_to_baseline_computes_deltas() -> None:
    experiment = {"metrics": {"retrieval": {"recall_at_5": 0.75}, "ops": {}}}
    baseline = {"metrics": {"retrieval": {"recall_at_5": 0.70}, "ops": {}}}
    deltas = compare_to_baseline(experiment, baseline)
    assert deltas["retrieval.recall_at_5"] == pytest.approx(0.05)


def test_compare_to_baseline_missing_metric_is_none() -> None:
    experiment = {"metrics": {"retrieval": {"recall_at_5": 0.75}}}
    baseline = {"metrics": {"retrieval": {}}}
    deltas = compare_to_baseline(experiment, baseline)
    assert deltas["retrieval.recall_at_5"] is None
