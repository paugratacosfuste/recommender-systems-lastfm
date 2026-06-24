"""The evaluation harness: score any recommender on a held-out split.

Every method is evaluated the same way - fit on ``train``, then for each test user
compare its top-``k`` recommendations against that user's held-out items. This is the
backbone of the project's method-comparison story (the 30% evaluation grade).
"""

from __future__ import annotations

import pandas as pd

from recsys.config import ITEM_COL, USER_COL
from recsys.eval.metrics import precision_at_k
from recsys.models.base import BaseRecommender


def evaluate(
    model: BaseRecommender,
    train: pd.DataFrame,
    test: pd.DataFrame,
    k: int = 10,
) -> dict[str, float]:
    """Fit ``model`` on ``train`` and average Precision@K over test users.

    Parameters
    ----------
    model : BaseRecommender
        An unfitted (or refittable) recommender.
    train : pandas.DataFrame
        Training interactions with ``[user_id, artist_id, weight]``.
    test : pandas.DataFrame
        Held-out interactions; relevance is defined per user from this set.
    k : int
        Cutoff rank for the metric.

    Returns
    -------
    dict
        ``{"precision_at_k": float, "k": k, "n_users": int}``.
    """
    model.fit(train)
    relevant_by_user = test.groupby(USER_COL)[ITEM_COL].apply(set).to_dict()

    scores: list[float] = []
    for user_id, relevant in relevant_by_user.items():
        recommended = model.recommend(user_id, n=k, exclude_seen=True)
        scores.append(precision_at_k(recommended, relevant, k))

    mean_precision = sum(scores) / len(scores) if scores else 0.0
    return {
        "precision_at_k": mean_precision,
        "k": float(k),
        "n_users": float(len(scores)),
    }
