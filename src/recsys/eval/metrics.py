"""Ranking metrics for implicit-feedback recommendation.

Phase 0 introduced Precision@K; Module 3 added Recall@K; Module 4 adds MAP (via
Average Precision) and NDCG. Beyond-accuracy metrics come in later modules.
"""

from __future__ import annotations

import math
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


def recall_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Fraction of the user's relevant items that appear in the top ``k``.

    Recall@K = (# relevant items among the first k recommended) / (# relevant items)

    Where Precision@K asks "how clean is my shortlist", Recall@K asks "how much of what
    the user wanted did I manage to surface". With only ``k`` slots, a user with many
    held-out items caps recall well below 1.0.

    Parameters
    ----------
    recommended : sequence of int
        Recommended item ids, ordered most- to least-relevant.
    relevant : iterable of int
        The set of items that are actually relevant (the held-out items).
    k : int
        Cutoff rank (must be > 0).

    Returns
    -------
    float
        Recall@K in [0, 1]. Returns 0.0 when the user has no relevant items.
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer; got {k}")
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    hits = sum(1 for item in recommended[:k] if item in relevant_set)
    return hits / len(relevant_set)


def average_precision_at_k(
    recommended: Sequence[int], relevant: Iterable[int], k: int
) -> float:
    """Average Precision@K - precision recomputed at each hit, then averaged.

    Unlike plain precision, AP rewards ranking relevant items *earlier*: a hit at rank 1
    contributes more than a hit at rank 5. Averaging AP over all users gives MAP (Mean
    Average Precision).

    Normalised by ``min(k, #relevant)`` so a perfect ranking scores 1.0.

    Returns
    -------
    float
        Average Precision@K in [0, 1]; 0.0 when the user has no relevant items.
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer; got {k}")
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    score = 0.0
    hits = 0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in relevant_set:
            hits += 1
            score += hits / rank
    return score / min(len(relevant_set), k)


def ndcg_at_k(recommended: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Normalised Discounted Cumulative Gain at K (binary relevance).

    Each hit contributes ``1 / log2(rank + 1)``, so hits near the top count for more. The
    score is normalised by the best achievable ordering (all relevant items first), giving
    a value in [0, 1] comparable across users with different numbers of relevant items.

    Returns
    -------
    float
        NDCG@K in [0, 1]; 0.0 when the user has no relevant items.
    """
    if k <= 0:
        raise ValueError(f"k must be a positive integer; got {k}")
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(recommended[:k], start=1)
        if item in relevant_set
    )
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
