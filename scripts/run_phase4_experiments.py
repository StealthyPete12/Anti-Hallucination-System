"""Runs the full Phase 4 experiment battery this environment can measure for
real (retrieval-only: no LLM provider key or running Qdrant server needed -
see eval/experiment.py's module docstring) and writes each result under
``experiments/<name>/``.

    uv run python -m scripts.run_phase4_experiments

Generator/reranker-on-faithfulness/query-rewriting/judge-agreement
experiments that need a real LLM call are intentionally not included here -
see the README's Phase 4 section for which numbers are measured vs.
documented as "requires live LLM credentials".
"""

from __future__ import annotations

import logging

from eval.experiment import ExperimentSpec, run_experiment

logger = logging.getLogger("scripts.run_phase4_experiments")

#: In-memory Qdrant + a fresh collection name per experiment, so no experiment's
#: index can bleed into another's, and no running Qdrant server is required.
BASE_OVERRIDES = {"qdrant_url": ":memory:"}

BATTERY: list[ExperimentSpec] = [
    ExperimentSpec(
        name="baseline_dense",
        description="Phase 1 baseline: dense-only retrieval, top_k=5, 512-token chunks.",
        overrides={**BASE_OVERRIDES, "qdrant_collection_name": "exp_baseline_dense"},
    ),
    ExperimentSpec(
        name="sparse_bm25",
        description="BM25 sparse-only retrieval (Phase 4 section 1).",
        overrides={
            **BASE_OVERRIDES,
            "qdrant_collection_name": "exp_sparse_bm25",
            "retrieval_mode": "sparse",
        },
    ),
    ExperimentSpec(
        name="hybrid_rrf",
        description="Dense + BM25 sparse, fused with Reciprocal Rank Fusion (Phase 4 section 1).",
        overrides={
            **BASE_OVERRIDES,
            "qdrant_collection_name": "exp_hybrid_rrf",
            "retrieval_mode": "hybrid",
        },
    ),
    ExperimentSpec(
        name="reranker_minilm",
        description="Dense retrieval (top 20) -> cross-encoder/ms-marco-MiniLM-L-6-v2 -> top 5 "
        "(Phase 4 section 2).",
        overrides={
            **BASE_OVERRIDES,
            "qdrant_collection_name": "exp_reranker_minilm",
            "reranker_enabled": True,
            "retrieval_fanout": 20,
        },
    ),
    ExperimentSpec(
        name="chunk_1024_overlap15",
        description="Chunk size 1024 tokens, overlap ratio unchanged at 15% (Phase 4 section 4).",
        overrides={
            **BASE_OVERRIDES,
            "qdrant_collection_name": "exp_chunk_1024_overlap15",
            "chunk_size_tokens": 1024,
        },
    ),
    ExperimentSpec(
        name="chunk_512_overlap0",
        description="Chunk size 512 (baseline), overlap ratio 0% (Phase 4 section 4).",
        overrides={
            **BASE_OVERRIDES,
            "qdrant_collection_name": "exp_chunk_512_overlap0",
            "chunk_overlap_ratio": 0.0,
        },
    ),
    ExperimentSpec(
        name="chunk_512_overlap20",
        description="Chunk size 512 (baseline), overlap ratio 20% (Phase 4 section 4).",
        overrides={
            **BASE_OVERRIDES,
            "qdrant_collection_name": "exp_chunk_512_overlap20",
            "chunk_overlap_ratio": 0.20,
        },
    ),
    ExperimentSpec(
        name="chunk_1024_overlap0",
        description="Chunk size 1024, overlap ratio 0% (Phase 4 section 4).",
        overrides={
            **BASE_OVERRIDES,
            "qdrant_collection_name": "exp_chunk_1024_overlap0",
            "chunk_size_tokens": 1024,
            "chunk_overlap_ratio": 0.0,
        },
    ),
    ExperimentSpec(
        name="chunk_1024_overlap20",
        description="Chunk size 1024, overlap ratio 20% (Phase 4 section 4).",
        overrides={
            **BASE_OVERRIDES,
            "qdrant_collection_name": "exp_chunk_1024_overlap20",
            "chunk_size_tokens": 1024,
            "chunk_overlap_ratio": 0.20,
        },
    ),
    ExperimentSpec(
        name="top_k_3",
        description="top_k=3 (Phase 4 section 5).",
        overrides={**BASE_OVERRIDES, "qdrant_collection_name": "exp_top_k_3", "top_k": 3},
    ),
    ExperimentSpec(
        name="top_k_10",
        description="top_k=10 (Phase 4 section 5).",
        overrides={**BASE_OVERRIDES, "qdrant_collection_name": "exp_top_k_10", "top_k": 10},
    ),
]


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    )
    results = {}
    for spec in BATTERY:
        logger.info("=== Running experiment: %s ===", spec.name)
        record = run_experiment(spec)
        results[spec.name] = record["metrics"]
        logger.info("=== %s: %s ===", spec.name, record["metrics"]["retrieval"])

    print("\n\n=== SUMMARY ===")
    for name, metrics in results.items():
        print(f"{name}: {metrics['retrieval']}")


if __name__ == "__main__":
    main()
