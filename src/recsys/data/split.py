"""Per-user train/test splitting for implicit-feedback evaluation.

With implicit feedback we evaluate ranking quality: for each user we hold out some of
their interactions and check whether a model ranks them highly. A per-user (rather than
global) split guarantees every test user is also seen in training, which a recommender
needs in order to produce personalised recommendations for them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.config import DEFAULT_SEED, USER_COL


def leave_n_out_split(
    interactions: pd.DataFrame,
    test_frac: float = 0.2,
    min_train: int = 1,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Randomly hold out a fraction of each user's interactions for testing.

    For every user, ``ceil(test_frac * n_interactions)`` interactions are moved to the
    test set, while always leaving at least ``min_train`` interactions in training.
    Users with too few interactions to satisfy ``min_train`` contribute only to training.

    Parameters
    ----------
    interactions : pandas.DataFrame
        Must contain a ``user_id`` column (plus item/weight columns, preserved as-is).
    test_frac : float
        Fraction of each user's interactions to hold out (0 < test_frac < 1).
    min_train : int
        Minimum interactions kept in training per user.
    seed : int
        Seed for reproducible shuffling.

    Returns
    -------
    (train, test) : tuple of pandas.DataFrame
        Disjoint subsets of ``interactions`` with the original columns and dtypes.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1); got {test_frac}")
    if USER_COL not in interactions.columns:
        raise ValueError(f"interactions must contain a '{USER_COL}' column")

    rng = np.random.default_rng(seed)
    test_indices: list[int] = []

    for _, group in interactions.groupby(USER_COL, sort=False):
        n = len(group)
        n_test = int(np.ceil(test_frac * n))
        # Never strip a user below min_train interactions in the training set.
        n_test = min(n_test, max(0, n - min_train))
        if n_test == 0:
            continue
        chosen = rng.choice(group.index.to_numpy(), size=n_test, replace=False)
        test_indices.extend(chosen.tolist())

    test_mask = interactions.index.isin(test_indices)
    test = interactions[test_mask].reset_index(drop=True)
    train = interactions[~test_mask].reset_index(drop=True)
    return train, test
