"""Deterministic retrieval-quality metrics: Recall@K, MRR, nDCG@K.

Pure Python, no I/O, no LLM calls. Every function takes two ordered lists of
opaque chunk-id strings - ``retrieved_chunk_ids`` (rank-ordered, as returned
by the vector store) and ``relevant_chunk_ids`` (the set of chunk ids that
would correctly support the answer) - and returns a score with no side
effects. This makes them instant to run and trivial to unit test with
hand-worked examples, independent of whatever produced the ids.

A ``None`` return means "not applicable": when ``relevant_chunk_ids`` is
empty there is nothing to recall (e.g. an ``unanswerable`` golden-set
question has no supporting chunks), so the metric is undefined rather than
zero. Callers should exclude ``None`` results from aggregate averages
instead of treating them as failures.
"""

from __future__ import annotations

import math


def recall_at_k(
    retrieved_chunk_ids: list[str], relevant_chunk_ids: list[str], k: int
) -> float | None:
    """1.0 if at least one relevant chunk appears in the top ``k`` retrieved results, else 0.0.

    This is "hit rate@k" (binary), not the fraction of relevant chunks
    retrieved - matching the golden set's use of ``supporting_chunk_ids`` as
    "any one of these would answer the question," not "all of these must be
    retrieved."
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not relevant_chunk_ids:
        return None
    relevant = set(relevant_chunk_ids)
    top_k = retrieved_chunk_ids[:k]
    return 1.0 if any(cid in relevant for cid in top_k) else 0.0


def mrr(retrieved_chunk_ids: list[str], relevant_chunk_ids: list[str]) -> float | None:
    """1 / rank of the first relevant chunk in ``retrieved_chunk_ids`` (rank is 1-indexed).

    0.0 if no relevant chunk appears anywhere in the retrieved list.
    """
    if not relevant_chunk_ids:
        return None
    relevant = set(relevant_chunk_ids)
    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def _dcg(relevances: list[float]) -> float:
    # Standard binary-relevance DCG: rel_i discounted by log2(rank + 1),
    # with rank 1-indexed (so position 1 -> log2(2) == 1, no discount).
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(
    retrieved_chunk_ids: list[str], relevant_chunk_ids: list[str], k: int = 10
) -> float | None:
    """Normalized Discounted Cumulative Gain at rank ``k``, with binary relevance.

    DCG@k = sum_{i=1}^{k} rel_i / log2(i + 1)
    IDCG@k = DCG@k of the ideal ranking (all relevant chunks first, up to k)
    nDCG@k = DCG@k / IDCG@k
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not relevant_chunk_ids:
        return None
    relevant = set(relevant_chunk_ids)

    top_k = retrieved_chunk_ids[:k]
    relevances = [1.0 if cid in relevant else 0.0 for cid in top_k]
    dcg = _dcg(relevances)

    ideal_relevant_count = min(len(relevant), k)
    idcg = _dcg([1.0] * ideal_relevant_count)
    if idcg == 0.0:
        return None

    return dcg / idcg
