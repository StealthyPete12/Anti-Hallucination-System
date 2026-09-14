"""Reciprocal Rank Fusion (RRF): combines multiple rank-ordered id lists
(e.g. a dense-retrieval ranking and a sparse/BM25 ranking) into one fused
ranking, without needing the two rankings' raw scores to be comparable.

RRF only looks at *rank*, not score magnitude - this is what makes it safe
to combine a cosine-similarity dense ranking with a BM25 sparse ranking,
whose score scales have nothing to do with each other.
"""

from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    rankings: list[list[str]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse several rank-ordered id lists into one, sorted by fused score.

    ``score(id) = sum over rankings containing id of 1 / (k + rank)``, where
    ``rank`` is 1-indexed. An id missing from a ranking simply contributes
    nothing from that ranking. Higher fused score is better.
    """
    scores: defaultdict[str, float] = defaultdict(float)

    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)

    return sorted(
        scores.items(),
        key=lambda kv: -kv[1],
    )
