"""The evaluation harness: score any recommender on a held-out split.

Every method is evaluated the same way - fit on ``train``, then for each test user
compare its top-``k`` recommendations against that user's held-out items. This is the
backbone of the project's method-comparison story (the 30% evaluation grade).
"""

from __future__ import annotations

import time

import pandas as pd

from recsys.config import ITEM_COL, USER_COL
from recsys.eval.beyond_accuracy import (
    BeyondAccuracyInputs,
    catalogue_coverage,
    intra_list_diversity,
    novelty,
)
from recsys.eval.metrics import (
    average_precision_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from recsys.models.base import BaseRecommender


def evaluate(
    model: BaseRecommender,
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 10,
    beyond: BeyondAccuracyInputs | None = None,
) -> dict[str, float]:
    """Fit ``model`` on ``train`` and average ranking metrics over test users.

    Parameters
    ----------
    model : BaseRecommender
        An unfitted (or refittable) recommender.
    train : pandas.DataFrame
        Training interactions with ``[user_id, artist_id, weight]``.
    test : pandas.DataFrame
        Held-out interactions; relevance is defined per user from this set.
    k : int
        Cutoff rank for the metrics.
    beyond : BeyondAccuracyInputs, optional
        If given, also computes coverage, intra-list diversity, and novelty.

    Returns
    -------
    dict
        Always includes ``precision_at_k``, ``recall_at_k``, ``map_at_k``, ``ndcg_at_k``,
        ``k``, ``n_users``, ``fit_seconds``. When ``beyond`` is given, also includes
        ``coverage``, ``diversity``, ``novelty``.
    """
    start = time.perf_counter()
    model.fit(train)
    fit_seconds = time.perf_counter() - start

    relevant_by_user = test.groupby(USER_COL)[ITEM_COL].apply(set).to_dict()

    precisions: list[float] = []
    recalls: list[float] = []
    average_precisions: list[float] = []
    ndcgs: list[float] = []
    diversities: list[float] = []
    novelties: list[float] = []
    all_recommended: list[list[int]] = []

    for user_id, relevant in relevant_by_user.items():
        recommended = model.recommend(user_id, n=k, exclude_seen=True)
        precisions.append(precision_at_k(recommended, relevant, k))
        recalls.append(recall_at_k(recommended, relevant, k))
        average_precisions.append(average_precision_at_k(recommended, relevant, k))
        ndcgs.append(ndcg_at_k(recommended, relevant, k))
        if beyond is not None:
            all_recommended.append(recommended)
            diversities.append(
                intra_list_diversity(recommended, beyond.profiles, beyond.item_to_row)
            )
            novelties.append(novelty(recommended, beyond.popularity))

    n = len(precisions)

    def mean(values: list[float]) -> float:
        return sum(values) / n if n else 0.0

    result = {
        "precision_at_k": mean(precisions),
        "recall_at_k": mean(recalls),
        "map_at_k": mean(average_precisions),
        "ndcg_at_k": mean(ndcgs),
        "k": float(k),
        "n_users": float(n),
        "fit_seconds": fit_seconds,
    }
    if beyond is not None:
        result["coverage"] = catalogue_coverage(all_recommended, beyond.n_catalogue)
        result["diversity"] = mean(diversities)
        result["novelty"] = mean(novelties)
    return result


def compare_models(
    models: dict[str, BaseRecommender],
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 10,
    beyond: BeyondAccuracyInputs | None = None,
) -> pd.DataFrame:
    """Evaluate several named models on the same split and return a comparison table.

    Parameters
    ----------
    models : dict of str -> BaseRecommender
        Named, unfitted recommenders to compare.
    train, test : pandas.DataFrame
        The shared train/test split (fairness depends on this being identical).
    k : int
        Cutoff rank.

    Returns
    -------
    pandas.DataFrame
        One row per model, indexed by name, sorted by Precision@K descending.
    """
    rows = {
        name: evaluate(model, train, test, k=k, beyond=beyond)
        for name, model in models.items()
    }
    table = pd.DataFrame.from_dict(rows, orient="index")
    return table.sort_values("precision_at_k", ascending=False)
