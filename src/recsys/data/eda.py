"""Exploratory statistics for the interaction data.

Pure, tested functions that the EDA notebook (and the findings log) build on. Keeping the
maths here - rather than inline in a notebook - means it is version-controlled and unit
tested, and the notebook stays a thin presentation layer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.config import DEFAULT_SEED, ITEM_COL, USER_COL, WEIGHT_COL


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


def most_active_users(interactions: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """The ``n`` users with the most interactions (and their total plays).

    Returns
    -------
    pandas.DataFrame
        Columns ``[user_id, n_artists, total_plays]``, sorted by ``n_artists`` descending.
    """
    grouped = interactions.groupby(USER_COL)
    table = pd.DataFrame(
        {
            "n_artists": grouped.size(),
            "total_plays": grouped[WEIGHT_COL].sum(),
        }
    )
    return table.sort_values("n_artists", ascending=False).head(n).reset_index()


def interaction_quality(interactions: pd.DataFrame) -> dict[str, float]:
    """Basic data-quality summary of the interaction table.

    Returns counts of duplicate user-artist pairs and non-positive weights (both should be
    zero in clean data) plus the play-count range, which exposes the superfan outliers.
    """
    w = interactions[WEIGHT_COL]
    return {
        "n_rows": float(len(interactions)),
        "n_duplicate_pairs": float(interactions.duplicated([USER_COL, ITEM_COL]).sum()),
        "n_nonpositive_weight": float((w <= 0).sum()),
        "min_weight": float(w.min()),
        "median_weight": float(w.median()),
        "max_weight": float(w.max()),
    }


def tag_frequency(
    tagged_artists: pd.DataFrame, tags: pd.DataFrame | None = None, n: int = 20
) -> pd.DataFrame:
    """The ``n`` most-applied tags (optionally joined with their human-readable labels).

    Parameters
    ----------
    tagged_artists : pandas.DataFrame
        Columns ``[user_id, artist_id, tag_id]``.
    tags : pandas.DataFrame, optional
        ``[tag_id, tag_value]`` lookup; if given, a ``tag`` label column is added.
    n : int
        Number of top tags to return.
    """
    counts = tagged_artists["tag_id"].value_counts().head(n)
    out = counts.rename_axis("tag_id").reset_index(name="assignments")
    if tags is not None:
        labels = dict(zip(tags["tag_id"], tags["tag_value"]))
        out["tag"] = out["tag_id"].map(labels)
    return out


def tags_per_artist(tagged_artists: pd.DataFrame) -> pd.Series:
    """Number of distinct tags applied to each artist."""
    return tagged_artists.groupby(ITEM_COL)["tag_id"].nunique()


def friend_degree(friends: pd.DataFrame) -> pd.Series:
    """Number of friends per user (degree in the friendship graph)."""
    return friends.groupby(USER_COL).size()


def friend_listening_overlap(
    interactions: pd.DataFrame,
    friends: pd.DataFrame,
    n_random: int = 2000,
    seed: int = DEFAULT_SEED,
) -> dict[str, float]:
    """Compare listening overlap (Jaccard) of friend pairs vs random user pairs.

    Friends listening to more of the same artists than random pairs is the empirical
    justification for the social recommender. Returns the mean Jaccard similarity of the
    artist sets for friend pairs, for random pairs, and their ratio.
    """
    listened = interactions.groupby(USER_COL)[ITEM_COL].apply(set).to_dict()

    def jaccard(a: int, b: int) -> float:
        set_a, set_b = listened.get(a, set()), listened.get(b, set())
        union = set_a | set_b
        return len(set_a & set_b) / len(union) if union else 0.0

    edges = {
        tuple(sorted((int(u), int(f))))
        for u, f in zip(friends[USER_COL], friends["friend_id"])
        if u != f
    }
    friend_scores = [jaccard(a, b) for a, b in edges if a in listened and b in listened]
    friend_mean = float(np.mean(friend_scores)) if friend_scores else 0.0

    rng = np.random.default_rng(seed)
    users = list(listened)
    random_scores = [
        jaccard(*(users[i] for i in rng.choice(len(users), size=2, replace=False)))
        for _ in range(n_random)
    ]
    random_mean = float(np.mean(random_scores)) if random_scores else 0.0

    return {
        "friend_jaccard": friend_mean,
        "random_jaccard": random_mean,
        "ratio": friend_mean / random_mean if random_mean > 0 else float("inf"),
    }


def top_k_play_share(interactions: pd.DataFrame, k: int) -> float:
    """Share of all plays captured by the ``k`` most popular artists.

    A concrete way to state popularity bias: e.g. "the top 100 artists account for X%
    of all listening".
    """
    if k <= 0:
        raise ValueError("k must be positive")
    curve = popularity_curve(interactions)
    return float(curve.head(k).sum() / curve.sum())
