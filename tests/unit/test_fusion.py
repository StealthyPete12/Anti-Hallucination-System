"""Unit tests for Reciprocal Rank Fusion (rag/fusion.py)."""

from __future__ import annotations

from rag.fusion import reciprocal_rank_fusion


def test_single_ranking_preserves_order() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"]])
    assert [doc_id for doc_id, _ in fused] == ["a", "b", "c"]


def test_matches_hand_computed_scores_k60() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60)
    scores = dict(fused)
    # a: rank1 in list1 (1/61) + rank2 in list2 (1/62); b: rank2 (1/62) + rank1 (1/61)
    expected = 1 / 61 + 1 / 62
    assert scores["a"] == expected
    assert scores["b"] == expected


def test_agreement_across_rankings_outranks_single_top_rank() -> None:
    # "b" is #1 in both rankings; "a" is #1 in only one and absent from the other.
    fused = reciprocal_rank_fusion([["a", "x", "y"], ["b", "z", "w"], ["b", "a", "q"]])
    order = [doc_id for doc_id, _ in fused]
    assert order.index("b") < order.index("a")


def test_doc_missing_from_a_ranking_only_scores_from_rankings_it_appears_in() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["a"]], k=60)
    scores = dict(fused)
    assert scores["a"] == 1 / 61 + 1 / 61
    assert scores["b"] == 1 / 62


def test_empty_rankings_returns_empty() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_result_is_sorted_descending_by_score() -> None:
    fused = reciprocal_rank_fusion([["c", "b", "a"], ["c", "a", "b"]])
    scores = [score for _, score in fused]
    assert scores == sorted(scores, reverse=True)


def test_custom_k_changes_relative_weighting() -> None:
    ranking = [["a", "b", "c"]]
    low_k = dict(reciprocal_rank_fusion(ranking, k=1))
    high_k = dict(reciprocal_rank_fusion(ranking, k=1000))
    # A small k makes rank 1 dominate far more strongly than a large k does.
    assert (low_k["a"] / low_k["c"]) > (high_k["a"] / high_k["c"])
