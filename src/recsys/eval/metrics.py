"""Ranking metrics for implicit-feedback recommendation.

Phase 0 introduces Precision@K. Recall@K, MAP, NDCG, and beyond-accuracy metrics are
added in later modules and will live alongside this function.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def precision_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Fraction of the top-``k`` recommendations that are relevant.

    Precision@K = (# relevant items among the first k recommended) / k

    Parameters
    ----------
    recommended : sequence of int
        Recommended item ids, ordered most- to least-relevant.
    relevant : iterable of int
        The set of items that are actually relevant for this user (the held-out items).
    k : int
        Cutoff rank (must be > 0).

    Returns
    -------
    float
        Precision@K in [0, 1]. By convention we divide by ``k`` (not by the number of
        items returned), so a model that returns fewer than k items is penalised.
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer; got {k}")
    relevant_set = set(relevant)
    hits = sum(1 for item in recommended[:k] if item in relevant_set)
    return hits / k
