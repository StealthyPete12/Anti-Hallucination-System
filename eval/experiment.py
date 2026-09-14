"""Phase 4 experiment framework: reproducible, config-driven pipeline experiments.

"Measure before you believe" - every experiment here builds a completely
independent :class:`RagPipeline` from a base config plus overrides (a
different embedding model, chunk size, retrieval mode, reranker, ...),
re-indexes a fresh in-memory Qdrant collection so index-affecting settings
(chunking, embedding) are actually reflected in what gets searched, runs it
against the real golden set, and writes a self-contained JSON record under
``experiments/<name>/`` carrying the git SHA, timestamp, exact config, and
measured metrics - so any experiment can be reproduced or audited later
without re-deriving what was actually run.

Two evaluation modes:

- ``retrieval_only`` (default, and the only mode this environment can run
  for real): calls :meth:`RagPipeline.retrieve` directly - embed, retrieve,
  optionally rerank - and scores Recall@5/MRR/nDCG@10 against the golden
  set's ``supporting_chunk_ids``. No LLM call of any kind, so it needs no
  provider API key and no running Qdrant server.
- ``full``: delegates to :func:`eval.run.run_evaluation` for the complete
  Phase 2 scorecard (adds faithfulness/correctness/refusal/cost via the LLM
  judge and generator). Requires a configured LLM provider key.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.dependencies import build_pipeline
from config import Settings
from config import settings as base_settings
from eval.dataset import GoldenQuestion, load_golden_set
from eval.metrics.retrieval import mrr, ndcg_at_k, recall_at_k
from eval.run import NDCG_K, RECALL_K, build_config, run_evaluation
from rag.pipeline import RagPipeline

logger = logging.getLogger("eval.experiment")

EXPERIMENTS_DIR = Path("experiments")


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
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not determine git sha: %s", exc)
        return "unknown"


def utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class ExperimentSpec:
    """One reproducible experiment: a name, a rationale, and Settings overrides."""

    name: str
    description: str
    overrides: dict[str, Any] = field(default_factory=dict)
    mode: str = "retrieval_only"  # "retrieval_only" | "full"


def build_experiment_settings(overrides: dict[str, Any]) -> Settings:
    """Base settings (env/`.env`-derived) with ``overrides`` applied on top.

    Uses ``Settings(**{**base, **overrides})`` rather than ``model_copy``
    (which skips validation) so a bad override - an unsupported
    retrieval_mode, a negative chunk size - fails loudly at experiment build
    time instead of silently producing a broken pipeline.
    """
    merged = {**base_settings.model_dump(), **overrides}
    return Settings(**merged)


def _mean(values: list[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return sum(present) / len(present) if present else None


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    lower, upper = int(rank), min(int(rank) + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def run_retrieval_only(pipeline: RagPipeline, questions: list[GoldenQuestion]) -> dict[str, Any]:
    """Score Recall@5/MRR/nDCG@10 + retrieval latency, with no LLM call.

    Every question calls the exact same :meth:`RagPipeline.retrieve` the
    production ``query()`` path uses internally, so this measures the real
    retrieval stack (mode/reranker/rewriter included), just without paying
    for or requiring generation.
    """
    recall_values: list[float | None] = []
    mrr_values: list[float | None] = []
    ndcg_values: list[float | None] = []
    embed_latencies: list[float] = []
    retrieve_latencies: list[float] = []
    per_question: list[dict[str, Any]] = []

    for question in questions:
        retrieved, stages = pipeline.retrieve(question.question)
        retrieved_ids = [c.chunk_id for c in retrieved]
        recall = recall_at_k(retrieved_ids, question.supporting_chunk_ids, k=RECALL_K)
        rr = mrr(retrieved_ids, question.supporting_chunk_ids)
        ndcg = ndcg_at_k(retrieved_ids, question.supporting_chunk_ids, k=NDCG_K)
        recall_values.append(recall)
        mrr_values.append(rr)
        ndcg_values.append(ndcg)
        embed_latencies.append(stages["embed"])
        retrieve_latencies.append(stages["retrieve"] + stages["rerank"])
        per_question.append(
            {
                "id": question.id,
                "type": question.type,
                "retrieved_chunk_ids": retrieved_ids,
                "supporting_chunk_ids": question.supporting_chunk_ids,
                "recall_at_5": recall,
                "mrr": rr,
                "ndcg_at_10": ndcg,
            }
        )

    return {
        "retrieval": {
            "recall_at_5": _mean(recall_values),
            "mrr": _mean(mrr_values),
            "ndcg_at_10": _mean(ndcg_values),
        },
        "ops": {
            "mean_embed_latency_ms": _mean(embed_latencies),  # type: ignore[arg-type]
            "p95_retrieve_latency_ms": _percentile(retrieve_latencies, 0.95),
        },
        "num_questions": len(questions),
        "per_question": per_question,
    }


def run_experiment(
    spec: ExperimentSpec,
    *,
    dataset_path: str = "data/golden_set.jsonl",
    out_dir: str | Path = EXPERIMENTS_DIR,
    judge_model: str | None = None,
) -> dict[str, Any]:
    """Build, index, and evaluate one experiment; persist and return its record."""
    settings = build_experiment_settings(spec.overrides)
    pipeline = build_pipeline(settings)
    questions = load_golden_set(dataset_path)

    logger.info("experiment=%s mode=%s overrides=%s", spec.name, spec.mode, spec.overrides)
    indexed = pipeline.index_corpus(Path(settings.corpus_dir))
    logger.info("experiment=%s indexed %d chunk(s)", spec.name, indexed)

    if spec.mode == "retrieval_only":
        metrics = run_retrieval_only(pipeline, questions)
        config = build_config(judge_model or settings.judge_model, pipeline=pipeline)
    elif spec.mode == "full":
        scorecard = asyncio.run(
            run_evaluation(
                dataset_path=dataset_path,
                out_path=None,
                repeats=1,
                concurrency=10,
                judge_model=judge_model or settings.judge_model,
                pipeline=pipeline,
            )
        )
        metrics = {
            "retrieval": scorecard["retrieval"],
            "generation": scorecard["generation"],
            "safety": scorecard["safety"],
            "ops": scorecard["ops"],
            "num_questions": scorecard["num_questions"],
            "per_question": scorecard["per_question"],
        }
        config = scorecard["config"]
    else:
        raise ValueError(f"Unknown experiment mode {spec.mode!r}")

    record = {
        "experiment": spec.name,
        "description": spec.description,
        "mode": spec.mode,
        "git_sha": get_git_sha(),
        "timestamp": utc_now_iso(),
        "overrides": spec.overrides,
        "config": config,
        "metrics": {k: v for k, v in metrics.items() if k != "per_question"},
        "per_question": metrics.get("per_question", []),
    }

    out_path = Path(out_dir) / spec.name
    out_path.mkdir(parents=True, exist_ok=True)
    stamp = record["timestamp"].replace(":", "").replace("-", "")
    (out_path / f"{stamp}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    (out_path / "latest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    logger.info("experiment=%s wrote %s", spec.name, out_path / f"{stamp}.json")

    return record


def compare_to_baseline(
    experiment_record: dict[str, Any], baseline_record: dict[str, Any]
) -> dict[str, float | None]:
    """Delta of every retrieval/ops metric, experiment minus baseline."""
    deltas: dict[str, float | None] = {}
    for section in ("retrieval", "ops", "generation", "safety"):
        exp_section = experiment_record.get("metrics", {}).get(section, {})
        base_section = baseline_record.get("metrics", {}).get(section, {})
        for key, exp_value in exp_section.items():
            base_value = base_section.get(key)
            if exp_value is None or base_value is None:
                deltas[f"{section}.{key}"] = None
            else:
                deltas[f"{section}.{key}"] = float(exp_value) - float(base_value)
    return deltas


def _coerce(value: str) -> Any:
    """Best-effort str -> bool/int/float/str coercion for CLI --override values."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one reproducible Phase 4 experiment against the golden set."
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--description", default="")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Settings field override, e.g. --override retrieval_mode=hybrid. Repeatable.",
    )
    parser.add_argument("--mode", choices=["retrieval_only", "full"], default="retrieval_only")
    parser.add_argument("--dataset", default="data/golden_set.jsonl")
    parser.add_argument("--out-dir", default=str(EXPERIMENTS_DIR))
    parser.add_argument("--log-level", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, (args.log_level or base_settings.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    overrides: dict[str, Any] = {}
    for item in args.override:
        key, _, value = item.partition("=")
        overrides[key] = _coerce(value)

    spec = ExperimentSpec(
        name=args.name, description=args.description, overrides=overrides, mode=args.mode
    )
    record = run_experiment(spec, dataset_path=args.dataset, out_dir=args.out_dir)
    print(json.dumps(record["metrics"], indent=2))


if __name__ == "__main__":
    main()
