"""Phase 2 evaluation harness CLI: runs the production RAG pipeline against
the golden set, scores retrieval/faithfulness/correctness/refusal, and writes
a scorecard.

    uv run python -m eval.run \\
        --dataset data/golden_set.jsonl \\
        --out evals/results.json \\
        --repeats 1

Every question's production call (embed -> retrieve -> generate) goes
through ``pipeline.query`` unchanged - the same code path the FastAPI
``/query`` endpoint uses - so the harness measures the real system, not a
separate evaluation-only implementation. Because ``pipeline.query`` is
synchronous (blocking network/model calls), each question runs in a worker
thread via ``asyncio.to_thread`` behind a semaphore, while the three judge
calls per question run natively async and concurrently with each other.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.dependencies import get_pipeline
from config import settings
from eval.dataset import GoldenQuestion, load_golden_set
from eval.judge import LLMJudge, load_prompt
from eval.metrics.correctness import evaluate_correctness
from eval.metrics.faithfulness import evaluate_faithfulness
from eval.metrics.refusal import evaluate_refusal
from eval.metrics.retrieval import mrr, ndcg_at_k, recall_at_k
from eval.results import QuestionResult
from eval.scorecard import build_scorecard, compute_noise_floor
from eval.summary import render_summary
from rag.pipeline import RagPipeline

logger = logging.getLogger("eval.run")

RECALL_K = 5
NDCG_K = 10

#: Prompt files whose hashes are recorded on every scorecard, so a future
#: prompt edit is immediately visible as a hash change rather than silently
#: redefining what "faithfulness" or "correctness" means run to run.
JUDGE_PROMPTS = ("faithfulness_v1", "correctness_v1", "refusal_v1")


def get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return result.stdout.strip()
    except Exception as exc:  # noqa: BLE001 - not a git repo, git missing, detached, etc.
        logger.debug("Could not determine git sha: %s", exc)
        return "unknown"


def utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


async def evaluate_question(
    pipeline: RagPipeline, judge: LLMJudge, question: GoldenQuestion
) -> QuestionResult:
    """Run one golden-set question through the production pipeline and every
    applicable metric, never raising - a failure is captured on the result's
    ``error`` field so one bad question can't abort the whole run.
    """
    try:
        result = await asyncio.to_thread(pipeline.query, question.question)
    except Exception as exc:  # noqa: BLE001
        logger.error("question=%s pipeline_failed error=%s", question.id, exc)
        return QuestionResult(
            id=question.id,
            type=question.type,
            question=question.question,
            generated_answer="",
            reference_answer=question.reference_answer,
            retrieved_chunk_ids=[],
            retrieval_metrics={"recall_at_5": None, "mrr": None, "ndcg_at_10": None},
            faithfulness=None,
            correctness=None,
            refusal=None,
            latency_ms={},
            cost_usd=0.0,
            trace_id="",
            error=f"pipeline query failed: {exc}",
        )

    retrieval_metrics = {
        "recall_at_5": recall_at_k(
            result.retrieved_chunk_ids, question.supporting_chunk_ids, k=RECALL_K
        ),
        "mrr": mrr(result.retrieved_chunk_ids, question.supporting_chunk_ids),
        "ndcg_at_10": ndcg_at_k(
            result.retrieved_chunk_ids, question.supporting_chunk_ids, k=NDCG_K
        ),
    }
    retrieved_chunks = [(c.chunk_id, c.text) for c in result.retrieved_chunks]

    faithfulness_dict: dict[str, Any] | None = None
    correctness_dict: dict[str, Any] | None = None
    refusal_dict: dict[str, Any] | None = None
    judge_cost = 0.0
    error: str | None = None

    try:
        judge_calls: dict[str, Any] = {
            "faithfulness": evaluate_faithfulness(
                judge, answer=result.answer, retrieved_chunks=retrieved_chunks
            ),
            "correctness": evaluate_correctness(
                judge,
                question=question.question,
                reference_answer=question.reference_answer,
                generated_answer=result.answer,
            ),
        }
        if question.requires_refusal:
            judge_calls["refusal"] = evaluate_refusal(
                judge,
                question=question.question,
                question_type=question.type,
                reference_answer=question.reference_answer,
                generated_answer=result.answer,
            )

        names = list(judge_calls.keys())
        outcomes = await asyncio.gather(*judge_calls.values())
        by_name = dict(zip(names, outcomes, strict=True))

        faithfulness_eval = by_name["faithfulness"]
        faithfulness_dict = {
            "score": faithfulness_eval.score,
            "unsupported_count": faithfulness_eval.unsupported_count,
            "claims": [
                {"text": c.text, "supported": c.supported, "chunk_id": c.chunk_id}
                for c in faithfulness_eval.claims
            ],
            "prompt_hash": faithfulness_eval.judge_response.prompt_hash,
        }
        judge_cost += faithfulness_eval.judge_response.cost_usd

        correctness_eval = by_name["correctness"]
        correctness_dict = {
            "verdict": correctness_eval.verdict,
            "reason": correctness_eval.reason,
            "score": correctness_eval.score,
            "prompt_hash": correctness_eval.judge_response.prompt_hash,
        }
        judge_cost += correctness_eval.judge_response.cost_usd

        if "refusal" in by_name:
            refusal_eval = by_name["refusal"]
            refusal_dict = {
                "refused": refusal_eval.refused,
                "reason": refusal_eval.reason,
                "prompt_hash": refusal_eval.judge_response.prompt_hash,
            }
            judge_cost += refusal_eval.judge_response.cost_usd
    except Exception as exc:  # noqa: BLE001
        logger.error("question=%s judge_failed error=%s", question.id, exc)
        error = f"judge evaluation failed: {exc}"

    return QuestionResult(
        id=question.id,
        type=question.type,
        question=question.question,
        generated_answer=result.answer,
        reference_answer=question.reference_answer,
        retrieved_chunk_ids=result.retrieved_chunk_ids,
        retrieval_metrics=retrieval_metrics,
        faithfulness=faithfulness_dict,
        correctness=correctness_dict,
        refusal=refusal_dict,
        latency_ms=result.latency_ms,
        cost_usd=result.cost_usd + judge_cost,
        trace_id=result.trace_id,
        error=error,
    )


async def run_pass(
    pipeline: RagPipeline,
    judge: LLMJudge,
    questions: list[GoldenQuestion],
    concurrency: int,
) -> list[QuestionResult]:
    """Evaluate every question, bounding in-flight pipeline+judge work to
    ``concurrency`` questions at a time."""
    semaphore = asyncio.Semaphore(concurrency)

    async def bound(question: GoldenQuestion) -> QuestionResult:
        async with semaphore:
            return await evaluate_question(pipeline, judge, question)

    return await asyncio.gather(*(bound(q) for q in questions))


def build_config(judge_model: str) -> dict[str, Any]:
    return {
        "embedder": settings.embedding_model_name,
        "generator": settings.llm_model,
        "judge": judge_model,
        "judge_prompt_hash": {name: load_prompt(name).sha256 for name in JUDGE_PROMPTS},
        "top_k": settings.top_k,
        "rerank": False,
    }


async def run_evaluation(
    *,
    dataset_path: str,
    out_path: str,
    repeats: int,
    concurrency: int,
    judge_model: str,
) -> dict[str, Any]:
    questions = load_golden_set(dataset_path)
    logger.info(
        "dataset=%s num_questions=%d repeats=%d concurrency=%d judge_model=%s",
        dataset_path,
        len(questions),
        repeats,
        concurrency,
        judge_model,
    )

    pipeline = get_pipeline()
    judge = LLMJudge(
        model=judge_model,
        timeout=settings.judge_timeout_seconds,
        max_retries=settings.judge_max_retries,
    )
    git_sha = get_git_sha()
    config = build_config(judge_model)
    logger.info("git_sha=%s judge_prompt_hash=%s", git_sha, config["judge_prompt_hash"])

    scorecards: list[dict[str, Any]] = []
    for pass_num in range(1, repeats + 1):
        t0 = time.perf_counter()
        results = await run_pass(pipeline, judge, questions, concurrency)
        elapsed_s = time.perf_counter() - t0
        num_errors = sum(1 for r in results if r.error is not None)
        logger.info(
            "pass=%d/%d elapsed_s=%.1f num_questions=%d num_errors=%d",
            pass_num,
            repeats,
            elapsed_s,
            len(results),
            num_errors,
        )
        scorecard = build_scorecard(
            results, git_sha=git_sha, timestamp=utc_now_iso(), config=config
        )
        scorecards.append(scorecard)

    if repeats == 1:
        output = scorecards[0]
    else:
        output = dict(scorecards[-1])
        output["repeats"] = repeats
        output["runs"] = [
            {
                "retrieval": sc["retrieval"],
                "generation": sc["generation"],
                "safety": sc["safety"],
                "ops": sc["ops"],
            }
            for sc in scorecards
        ]
        output["noise_floor"] = compute_noise_floor(scorecards)

    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
    logger.info("Wrote scorecard to %s", out_file)

    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Phase 2 evaluation harness against the golden set."
    )
    parser.add_argument("--dataset", default="data/golden_set.jsonl")
    parser.add_argument("--out", default="evals/results.json")
    parser.add_argument(
        "--repeats", type=int, default=1, help="Full evaluation passes; >1 computes a noise floor."
    )
    parser.add_argument(
        "--concurrency", type=int, default=10, help="Max concurrent in-flight questions."
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="LiteLLM model string for the judge (default: settings.judge_model)",
    )
    parser.add_argument("--log-level", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, (args.log_level or settings.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    judge_model = args.judge_model or settings.judge_model
    output = asyncio.run(
        run_evaluation(
            dataset_path=args.dataset,
            out_path=args.out,
            repeats=args.repeats,
            concurrency=args.concurrency,
            judge_model=judge_model,
        )
    )
    print(render_summary(output, noise_floor=output.get("noise_floor")))


if __name__ == "__main__":
    main()
