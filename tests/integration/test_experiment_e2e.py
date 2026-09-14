"""End-to-end integration test for the Phase 4 experiment framework
(eval/experiment.py): real chunker + real embedder + real in-memory Qdrant,
exercised through the same ``run_experiment`` entry point
``scripts/run_phase4_experiments.py`` uses for the real measured battery.

No LLM call is made (``mode="retrieval_only"``, same reasoning as
test_pipeline_e2e.py: no provider API key available in CI). Skipped
automatically if the embedding model can't be downloaded (e.g. no network).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval.experiment import ExperimentSpec, run_experiment
from rag.embedders.local import EmbeddingError, LocalBGEEmbedder

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def _embedder_available() -> None:
    embedder = LocalBGEEmbedder()
    try:
        _ = embedder.dimension  # forces model load
    except EmbeddingError as exc:
        pytest.skip(f"Could not load embedding model (offline?): {exc}")


@pytest.fixture
def tiny_corpus_and_dataset(tmp_path: Path) -> tuple[Path, Path]:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "postgres_mvcc.txt").write_text(
        "MVCC stands for Multiversion Concurrency Control. It lets readers "
        "and writers avoid blocking each other in PostgreSQL.",
        encoding="utf-8",
    )
    (corpus_dir / "postgres_indexes.txt").write_text(
        "By default, CREATE INDEX builds a B-tree index, which handles "
        "equality and range queries efficiently.",
        encoding="utf-8",
    )

    dataset_path = tmp_path / "golden_set.jsonl"
    questions = [
        {
            "id": "q1",
            "type": "single_hop",
            "question": "What does MVCC stand for?",
            "reference_answer": "Multiversion Concurrency Control",
            "supporting_chunk_ids": ["postgres_mvcc#0"],
            "tags": [],
        },
        {
            "id": "q2",
            "type": "single_hop",
            "question": "What kind of index does CREATE INDEX build by default?",
            "reference_answer": "A B-tree index",
            "supporting_chunk_ids": ["postgres_indexes#0"],
            "tags": [],
        },
    ]
    with dataset_path.open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")

    return corpus_dir, dataset_path


def test_run_experiment_retrieval_only_end_to_end(
    _embedder_available: None, tiny_corpus_and_dataset: tuple[Path, Path]
) -> None:
    corpus_dir, dataset_path = tiny_corpus_and_dataset
    spec = ExperimentSpec(
        name="e2e_test_dense",
        description="e2e smoke test",
        overrides={
            "qdrant_url": ":memory:",
            "corpus_dir": str(corpus_dir),
            "chunk_size_tokens": 64,
        },
        mode="retrieval_only",
    )

    record = run_experiment(spec, dataset_path=str(dataset_path), out_dir="/tmp/phase4_e2e_test")

    assert record["experiment"] == "e2e_test_dense"
    assert record["mode"] == "retrieval_only"
    assert record["metrics"]["num_questions"] == 2
    # The real embedder should surface each question's actually-relevant chunk.
    assert record["metrics"]["retrieval"]["recall_at_5"] == 1.0
    assert set(record["config"]) >= {"embedder", "top_k", "retrieval_mode"}


def test_run_experiment_hybrid_mode_end_to_end(
    _embedder_available: None, tiny_corpus_and_dataset: tuple[Path, Path]
) -> None:
    corpus_dir, dataset_path = tiny_corpus_and_dataset
    spec = ExperimentSpec(
        name="e2e_test_hybrid",
        description="e2e hybrid smoke test",
        overrides={
            "qdrant_url": ":memory:",
            "corpus_dir": str(corpus_dir),
            "chunk_size_tokens": 64,
            "retrieval_mode": "hybrid",
        },
        mode="retrieval_only",
    )

    record = run_experiment(spec, dataset_path=str(dataset_path), out_dir="/tmp/phase4_e2e_test")

    assert record["config"]["retrieval_mode"] == "hybrid"
    assert record["metrics"]["retrieval"]["recall_at_5"] == 1.0
