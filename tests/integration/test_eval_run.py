"""Integration tests for the eval.run orchestration: dataset loading, the
async per-question pipeline+judge fan-out, scorecard aggregation, noise-floor
computation over --repeats, and writing the scorecard to disk.

Both the RagPipeline and the judge are fakes (no Qdrant, no embedding model,
no LLM provider) - this test's job is to prove the modules wired together in
eval/run.py behave correctly as a whole, not to exercise any single one of
them again (that's what tests/unit/ already does).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eval import run as eval_run
from eval.judge import JudgeResponse
from rag.models import PipelineResult, RetrievedChunk, TokenUsage

pytestmark = pytest.mark.anyio


def make_retrieved(chunk_id: str) -> RetrievedChunk:
    doc_id = chunk_id.split("#")[0]
    return RetrievedChunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=f"text for {chunk_id}",
        score=0.9,
        source_path=f"{doc_id}.txt",
        heading="H",
        chunk_index=0,
    )


class FakePipeline:
    def __init__(self, responses: dict[str, PipelineResult]) -> None:
        self._responses = responses
        self.calls: list[str] = []

    def query(self, question: str, trace_id: str | None = None) -> PipelineResult:
        self.calls.append(question)
        return self._responses[question]


class FakeJudge:
    """Duck-types eval.judge.LLMJudge's constructor and .evaluate() signature."""

    def __init__(
        self, model: str | None = None, timeout: float | None = None, max_retries: int | None = None
    ) -> None:
        self.model = model
        self.calls: list[str] = []

    async def evaluate(self, prompt_name: str, variables: dict[str, str]) -> JudgeResponse:
        self.calls.append(prompt_name)
        if prompt_name == "faithfulness_v1":
            data: dict[str, Any] = {
                "claims": [
                    {"text": "claim", "supported": True, "chunk_id": "postgres_mvcc_intro#0"}
                ],
                "unsupported_count": 0,
            }
        elif prompt_name == "correctness_v1":
            data = {"verdict": "correct", "reason": "matches reference"}
        elif prompt_name == "refusal_v1":
            data = {"refused": True, "reason": "correctly declined"}
        else:
            raise ValueError(f"unexpected prompt {prompt_name!r}")
        return JudgeResponse(
            data=data,
            prompt_name=prompt_name,
            prompt_version="v1",
            prompt_hash="fakehash",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            cost_usd=0.0001,
            latency_ms=1.0,
            raw_content=str(data),
            attempts=1,
        )


@pytest.fixture
def golden_set_path(tmp_path: Path) -> Path:
    records = [
        {
            "id": "q001",
            "type": "single_hop",
            "question": "What is MVCC?",
            "reference_answer": "Multiversion Concurrency Control.",
            "supporting_chunk_ids": ["postgres_mvcc_intro#0"],
            "tags": ["mvcc"],
        },
        {
            "id": "q002",
            "type": "unanswerable",
            "question": "What is the max row count?",
            "reference_answer": (
                "The supplied context does not contain enough information to answer that question."
            ),
            "supporting_chunk_ids": [],
            "tags": ["limits"],
        },
    ]
    path = tmp_path / "golden.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def fake_pipeline() -> FakePipeline:
    return FakePipeline(
        {
            "What is MVCC?": PipelineResult(
                answer="MVCC stands for Multiversion Concurrency Control. [postgres_mvcc_intro#0]",
                citations=["postgres_mvcc_intro#0"],
                retrieved_chunk_ids=["postgres_mvcc_intro#0", "postgres_mvcc_caveats#1"],
                retrieved_chunks=[
                    make_retrieved("postgres_mvcc_intro#0"),
                    make_retrieved("postgres_mvcc_caveats#1"),
                ],
                latency_ms={"embed": 1.0, "retrieve": 1.0, "generate": 10.0, "total": 12.0},
                usage=TokenUsage(prompt_tokens=50, completion_tokens=10, total_tokens=60),
                cost_usd=0.0005,
                trace_id="t1",
            ),
            "What is the max row count?": PipelineResult(
                answer=(
                    "The supplied context does not contain enough information "
                    "to answer that question."
                ),
                citations=[],
                retrieved_chunk_ids=["postgres_ddl_basics#0"],
                retrieved_chunks=[make_retrieved("postgres_ddl_basics#0")],
                latency_ms={"embed": 1.0, "retrieve": 1.0, "generate": 8.0, "total": 10.0},
                usage=TokenUsage(prompt_tokens=40, completion_tokens=8, total_tokens=48),
                cost_usd=0.0003,
                trace_id="t2",
            ),
        }
    )


@pytest.fixture(autouse=True)
def patch_pipeline_and_judge(monkeypatch: pytest.MonkeyPatch, fake_pipeline: FakePipeline) -> None:
    monkeypatch.setattr(eval_run, "get_pipeline", lambda: fake_pipeline)
    monkeypatch.setattr(eval_run, "LLMJudge", FakeJudge)


async def test_run_evaluation_single_pass_writes_scorecard(
    golden_set_path: Path, tmp_path: Path
) -> None:
    out_path = tmp_path / "results.json"
    output = await eval_run.run_evaluation(
        dataset_path=str(golden_set_path),
        out_path=str(out_path),
        repeats=1,
        concurrency=5,
        judge_model="fake-model",
    )

    assert out_path.exists()
    on_disk = json.loads(out_path.read_text(encoding="utf-8"))
    assert on_disk == output

    assert output["num_questions"] == 2
    assert output["num_errors"] == 0
    assert len(output["per_question"]) == 2


async def test_run_evaluation_retrieval_metrics_computed_correctly(
    golden_set_path: Path, tmp_path: Path
) -> None:
    output = await eval_run.run_evaluation(
        dataset_path=str(golden_set_path),
        out_path=str(tmp_path / "out.json"),
        repeats=1,
        concurrency=5,
        judge_model="fake-model",
    )
    # q1's supporting chunk is retrieved at rank 1 -> perfect retrieval; q2 (unanswerable)
    # has no supporting_chunk_ids, so it's excluded from the retrieval average entirely.
    assert output["retrieval"]["recall_at_5"] == pytest.approx(1.0)
    assert output["retrieval"]["mrr"] == pytest.approx(1.0)
    assert output["retrieval"]["ndcg_at_10"] == pytest.approx(1.0)


async def test_run_evaluation_faithfulness_and_correctness_from_judge(
    golden_set_path: Path, tmp_path: Path
) -> None:
    output = await eval_run.run_evaluation(
        dataset_path=str(golden_set_path),
        out_path=str(tmp_path / "out.json"),
        repeats=1,
        concurrency=5,
        judge_model="fake-model",
    )
    assert output["generation"]["faithfulness"] == pytest.approx(1.0)
    assert output["generation"]["correctness"] == pytest.approx(1.0)


async def test_run_evaluation_refusal_only_scored_for_required_types(
    golden_set_path: Path, tmp_path: Path
) -> None:
    output = await eval_run.run_evaluation(
        dataset_path=str(golden_set_path),
        out_path=str(tmp_path / "out.json"),
        repeats=1,
        concurrency=5,
        judge_model="fake-model",
    )
    q1 = next(q for q in output["per_question"] if q["id"] == "q001")
    q2 = next(q for q in output["per_question"] if q["id"] == "q002")
    assert q1["refusal"] is None  # single_hop: refusal metric not applicable
    assert q2["refusal"] == {
        "refused": True,
        "reason": "correctly declined",
        "prompt_hash": "fakehash",
    }
    assert output["safety"]["refusal_rate"] == pytest.approx(1.0)
    assert output["safety"]["fabrications"] == 0


async def test_run_evaluation_cost_includes_generation_and_judge_cost(
    golden_set_path: Path, tmp_path: Path
) -> None:
    output = await eval_run.run_evaluation(
        dataset_path=str(golden_set_path),
        out_path=str(tmp_path / "out.json"),
        repeats=1,
        concurrency=5,
        judge_model="fake-model",
    )
    q1 = next(q for q in output["per_question"] if q["id"] == "q001")
    # generation 0.0005 + faithfulness 0.0001 + correctness 0.0001 (no refusal call for single_hop)
    assert q1["cost_usd"] == pytest.approx(0.0007)


async def test_run_evaluation_config_includes_prompt_hashes(
    golden_set_path: Path, tmp_path: Path
) -> None:
    output = await eval_run.run_evaluation(
        dataset_path=str(golden_set_path),
        out_path=str(tmp_path / "out.json"),
        repeats=1,
        concurrency=5,
        judge_model="fake-model",
    )
    assert set(output["config"]["judge_prompt_hash"]) == {
        "faithfulness_v1",
        "correctness_v1",
        "refusal_v1",
    }
    assert output["config"]["judge"] == "fake-model"


async def test_run_evaluation_repeats_produces_noise_floor(
    golden_set_path: Path, tmp_path: Path
) -> None:
    output = await eval_run.run_evaluation(
        dataset_path=str(golden_set_path),
        out_path=str(tmp_path / "out.json"),
        repeats=3,
        concurrency=5,
        judge_model="fake-model",
    )
    assert output["repeats"] == 3
    assert len(output["runs"]) == 3
    assert output["noise_floor"]["retrieval.recall_at_5"]["n"] == 3
    assert output["noise_floor"]["retrieval.recall_at_5"]["mean"] == pytest.approx(1.0)
    assert output["noise_floor"]["retrieval.recall_at_5"]["stdev"] == pytest.approx(0.0)


async def test_run_evaluation_single_repeat_has_no_noise_floor_key(
    golden_set_path: Path, tmp_path: Path
) -> None:
    output = await eval_run.run_evaluation(
        dataset_path=str(golden_set_path),
        out_path=str(tmp_path / "out.json"),
        repeats=1,
        concurrency=5,
        judge_model="fake-model",
    )
    assert "noise_floor" not in output
    assert "repeats" not in output


async def test_run_evaluation_bounds_concurrency(
    golden_set_path: Path, tmp_path: Path, fake_pipeline: FakePipeline
) -> None:
    """concurrency=1 must not change correctness, only how many run in flight at once."""
    output = await eval_run.run_evaluation(
        dataset_path=str(golden_set_path),
        out_path=str(tmp_path / "out.json"),
        repeats=1,
        concurrency=1,
        judge_model="fake-model",
    )
    assert output["num_questions"] == 2
    assert sorted(fake_pipeline.calls) == ["What is MVCC?", "What is the max row count?"]


def test_cli_main_writes_output_and_prints_summary(
    golden_set_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_path = tmp_path / "cli_results.json"
    eval_run.main(
        [
            "--dataset",
            str(golden_set_path),
            "--out",
            str(out_path),
            "--repeats",
            "1",
            "--concurrency",
            "4",
            "--judge-model",
            "fake-model",
        ]
    )
    assert out_path.exists()
    captured = capsys.readouterr()
    assert "Retrieval" in captured.out
    assert "Generation" in captured.out
    assert "Safety" in captured.out
    assert "Ops" in captured.out
