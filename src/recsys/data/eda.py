"""Exploratory statistics for the interaction data.

Pure, tested functions that the EDA notebook (and the findings log) build on. Keeping the
maths here - rather than inline in a notebook - means it is version-controlled and unit
tested, and the notebook stays a thin presentation layer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.config import ITEM_COL, USER_COL, WEIGHT_COL


def gini(values: np.ndarray | pd.Series) -> float:
    """Gini coefficient of a non-negative distribution.

    0.0 means perfectly equal (every artist equally popular); values near 1.0 mean
    extreme concentration (a few artists soak up almost all the plays). This is the
    headline number for the popularity-bias discussion.

    Parameters
    ----------
    values : array-like
        Non-negative quantities (e.g. plays or listeners per artist).

    Returns
    -------
    float
        Gini coefficient in [0, 1]. Returns 0.0 for an all-zero or empty input.
    """
    x = np.sort(np.asarray(values, dtype=float))
    if x.size == 0 or x.sum() == 0:
        return 0.0
    if (x < 0).any():
        raise ValueError("gini is only defined for non-negative values")
    n = x.size
    index = np.arange(1, n + 1)
    # Mean absolute difference formulation via the sorted "Lorenz" identity.
    return float((2.0 * np.sum(index * x)) / (n * np.sum(x)) - (n + 1.0) / n)


def summary_stats(interactions: pd.DataFrame) -> dict[str, float]:
    """Headline dataset statistics used in the EDA and findings log."""
    n_users = interactions[USER_COL].nunique()
    n_items = interactions[ITEM_COL].nunique()
    n_obs = len(interactions)
    density = n_obs / (n_users * n_items)
    per_user = interactions.groupby(USER_COL).size()
    per_item = interactions.groupby(ITEM_COL).size()
    return {
        "n_users": float(n_users),
        "n_items": float(n_items),
        "n_interactions": float(n_obs),
        "density": float(density),
        "sparsity": float(1.0 - density),
        "median_artists_per_user": float(per_user.median()),
        "median_listeners_per_artist": float(per_item.median()),
        "plays_gini": gini(interactions.groupby(ITEM_COL)[WEIGHT_COL].sum()),
    }


def popularity_curve(interactions: pd.DataFrame) -> pd.Series:
    """Total plays per artist, sorted descending (the long-tail curve)."""
    return (
        interactions.groupby(ITEM_COL)[WEIGHT_COL]
        .sum()
        .sort_values(ascending=False)
        .reset_index(drop=True)
    )


def top_k_play_share(interactions: pd.DataFrame, k: int) -> float:
    """Share of all plays captured by the ``k`` most popular artists.

    A concrete way to state popularity bias: e.g. "the top 100 artists account for X%
    of all listening".
    """
    if k <= 0:
        raise ValueError("k must be positive")
    curve = popularity_curve(interactions)
    return float(curve.head(k).sum() / curve.sum())
