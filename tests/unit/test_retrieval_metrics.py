"""Unit tests for eval/metrics/retrieval.py: Recall@K, MRR, nDCG@K.

Every expected value is manually derived from the standard IR formulas in
the docstrings of the functions under test, not just re-run through the
implementation - see the inline comments on each nDCG case.
"""

from __future__ import annotations

import math

import pytest

from eval.metrics.retrieval import mrr, ndcg_at_k, recall_at_k

# ---------------------------------------------------------------------------
# Recall@K
# ---------------------------------------------------------------------------


def test_recall_happy_path_relevant_chunk_in_top_k() -> None:
    assert recall_at_k(["a", "b", "c"], ["b"], k=3) == 1.0


def test_recall_relevant_chunk_outside_top_k() -> None:
    assert recall_at_k(["a", "b", "c", "d"], ["d"], k=2) == 0.0


def test_recall_multiple_relevant_chunks_only_one_needed() -> None:
    # b is relevant and within top 3; x is relevant but never retrieved - recall
    # is still 1.0 because at least one relevant chunk was surfaced.
    assert recall_at_k(["a", "b", "c"], ["b", "x"], k=3) == 1.0


def test_recall_missing_relevant_chunk_entirely() -> None:
    assert recall_at_k(["a", "c", "d"], ["b"], k=3) == 0.0


def test_recall_k_smaller_than_retrieved_list() -> None:
    # Relevant chunk "c" is retrieved, but at rank 3, outside k=2.
    assert recall_at_k(["a", "b", "c"], ["c"], k=2) == 0.0


def test_recall_empty_relevant_set_is_not_applicable() -> None:
    assert recall_at_k(["a", "b"], [], k=5) is None


def test_recall_empty_retrieved_list() -> None:
    assert recall_at_k([], ["a"], k=5) == 0.0


def test_recall_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError):
        recall_at_k(["a"], ["a"], k=0)


# ---------------------------------------------------------------------------
# MRR
# ---------------------------------------------------------------------------


def test_mrr_first_result_is_relevant() -> None:
    assert mrr(["a", "b", "c"], ["a"]) == 1.0


def test_mrr_relevant_chunk_at_rank_three() -> None:
    assert mrr(["a", "b", "c"], ["c"]) == pytest.approx(1 / 3)


def test_mrr_multiple_relevant_chunks_uses_earliest_rank() -> None:
    # "a" (rank 2) and "c" (rank 3) are both relevant; MRR uses the earliest.
    assert mrr(["x", "a", "c"], ["c", "a"]) == pytest.approx(1 / 2)


def test_mrr_no_relevant_chunk_found() -> None:
    assert mrr(["a", "b", "c"], ["z"]) == 0.0


def test_mrr_empty_relevant_set_is_not_applicable() -> None:
    assert mrr(["a", "b"], []) is None


def test_mrr_empty_retrieved_list() -> None:
    assert mrr([], ["a"]) == 0.0


# ---------------------------------------------------------------------------
# nDCG@K
# ---------------------------------------------------------------------------


def test_ndcg_perfect_ranking_scores_one() -> None:
    # Both relevant chunks ("a", "b") occupy the top two ranks - the ideal
    # ranking - so actual DCG equals ideal DCG (IDCG) exactly.
    assert ndcg_at_k(["a", "b", "c"], ["a", "b"], k=3) == pytest.approx(1.0)


def test_ndcg_single_relevant_chunk_at_worst_rank() -> None:
    # relevance vector at ranks 1..3 is [0, 0, 1].
    # DCG = 1 / log2(3 + 1) = 1 / 2 = 0.5
    # IDCG (1 relevant chunk, ideal rank 1) = 1 / log2(1 + 1) = 1 / 1 = 1.0
    # nDCG = 0.5 / 1.0 = 0.5
    assert ndcg_at_k(["a", "b", "c"], ["c"], k=3) == pytest.approx(0.5)


def test_ndcg_multiple_relevant_chunks_mixed_order() -> None:
    # relevant = {b, d}; retrieved = [a, b, c, d, e] -> relevance vector [0,1,0,1,0]
    # DCG = 1/log2(3) + 1/log2(5)   (ranks 2 and 4, 1-indexed -> log2(rank+1))
    # IDCG (2 relevant chunks, ideal ranks 1 and 2) = 1/log2(2) + 1/log2(3)
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3)
    expected = dcg / idcg
    assert ndcg_at_k(["a", "b", "c", "d", "e"], ["b", "d"], k=5) == pytest.approx(expected)


def test_ndcg_relevant_chunk_beyond_k_is_a_zero() -> None:
    # "b" is relevant but sits at rank 5, outside k=3, so it never contributes.
    assert ndcg_at_k(["a", "x", "y", "z", "b"], ["b"], k=3) == pytest.approx(0.0)


def test_ndcg_more_relevant_chunks_than_retrieved() -> None:
    # relevant has 3 members but only 2 chunks were ever retrieved.
    # DCG = 1/log2(2) + 1/log2(3)  (both retrieved chunks are relevant)
    # IDCG uses min(len(relevant), k) = min(3, 5) = 3 ideal ranks.
    dcg = 1 / math.log2(2) + 1 / math.log2(3)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    expected = dcg / idcg
    assert ndcg_at_k(["a", "c"], ["a", "b", "c"], k=5) == pytest.approx(expected)


def test_ndcg_missing_relevant_chunk_entirely_scores_zero() -> None:
    assert ndcg_at_k(["a", "b", "c"], ["z"], k=3) == pytest.approx(0.0)


def test_ndcg_empty_relevant_set_is_not_applicable() -> None:
    assert ndcg_at_k(["a", "b"], [], k=10) is None


def test_ndcg_empty_retrieved_list() -> None:
    assert ndcg_at_k([], ["a"], k=10) == pytest.approx(0.0)


def test_ndcg_default_k_is_ten() -> None:
    retrieved = [f"c{i}" for i in range(15)]
    # relevant chunk at rank 11 is outside the default k=10 cutoff.
    assert ndcg_at_k(retrieved, ["c10"]) == pytest.approx(0.0)


def test_ndcg_rejects_non_positive_k() -> None:
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], ["a"], k=0)
