"""Turn tidy interactions into the structures the personalised models need.

Two things every collaborative / matrix-factorisation model requires:

1. A mapping from the dataset's arbitrary ``user_id`` / ``artist_id`` values to
   contiguous integer indices ``0..n-1``, so they can index rows/columns of a matrix.
2. A sparse user-item matrix whose stored values are *confidence weights* derived from
   the raw play counts (implicit feedback), not the raw counts themselves.

Confidence weighting follows Hu, Koren & Volinsky (2008): we are far more confident that
a user likes an artist they played 5,000 times than one they played twice, but the
relationship is sub-linear, so we log-scale before weighting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from recsys.config import ITEM_COL, USER_COL, WEIGHT_COL


@dataclass(frozen=True)
class IndexMapping:
    """Bidirectional map between dataset ids and contiguous matrix indices.

    Attributes
    ----------
    user_ids : np.ndarray
        Sorted unique user ids; position == matrix row index.
    item_ids : np.ndarray
        Sorted unique artist ids; position == matrix column index.
    """

    user_ids: np.ndarray
    item_ids: np.ndarray

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    @property
    def _user_to_index(self) -> dict[int, int]:
        return {uid: i for i, uid in enumerate(self.user_ids)}

    @property
    def _item_to_index(self) -> dict[int, int]:
        return {iid: i for i, iid in enumerate(self.item_ids)}

    def user_index(self, user_id: int) -> int:
        """Matrix row index for a user id (raises KeyError if unknown)."""
        return self._user_to_index[user_id]

    def item_index(self, item_id: int) -> int:
        """Matrix column index for an artist id (raises KeyError if unknown)."""
        return self._item_to_index[item_id]


def build_index_mapping(interactions: pd.DataFrame) -> IndexMapping:
    """Create an :class:`IndexMapping` from the users and artists in ``interactions``."""
    user_ids = np.sort(interactions[USER_COL].unique())
    item_ids = np.sort(interactions[ITEM_COL].unique())
    return IndexMapping(user_ids=user_ids, item_ids=item_ids)


def log_scale(weights: np.ndarray | pd.Series) -> np.ndarray:
    """Compress raw play counts with ``log1p`` (= ``log(1 + x)``).

    Play counts span several orders of magnitude (1 to ~350k). Log-scaling stops a few
    superfans from dominating and better reflects diminishing marginal preference.
    """
    return np.log1p(np.asarray(weights, dtype=float))


def confidence(
    weights: np.ndarray | pd.Series, alpha: float = 40.0, eps: float = 1e-8
) -> np.ndarray:
    """Hu/Koren/Volinsky confidence weights from raw play counts.

    ``c = 1 + alpha * log(1 + r / eps)``

    Every observed interaction gets at least confidence 1; more plays raise it, but
    logarithmically. ``alpha`` controls how steeply confidence grows with plays.

    Parameters
    ----------
    weights : array-like
        Raw play counts (> 0).
    alpha : float
        Confidence scaling factor (tuned later for ALS in Module 6).
    eps : float
        Small constant inside the log.
    """
    r = np.asarray(weights, dtype=float)
    return 1.0 + alpha * np.log1p(r / eps)


def build_interaction_matrix(
    interactions: pd.DataFrame,
    mapping: IndexMapping,
    value_col: str = WEIGHT_COL,
) -> csr_matrix:
    """Build a sparse ``(n_users, n_items)`` matrix of ``value_col`` values.

    Parameters
    ----------
    interactions : pandas.DataFrame
        Interactions with ``[user_id, artist_id]`` and the chosen ``value_col``.
    mapping : IndexMapping
        Defines row/column ordering. Interactions for ids absent from the mapping are
        dropped (e.g. a held-out artist not present in training).
    value_col : str
        Which column to store as the matrix value (e.g. raw weight or a confidence/log
        column you added beforehand).

    Returns
    -------
    scipy.sparse.csr_matrix
        Compressed sparse row matrix, shape ``(n_users, n_items)``.
    """
    if value_col not in interactions.columns:
        raise ValueError(f"value_col {value_col!r} not in interactions columns")

    user_to_index = {uid: i for i, uid in enumerate(mapping.user_ids)}
    item_to_index = {iid: i for i, iid in enumerate(mapping.item_ids)}

    rows = interactions[USER_COL].map(user_to_index)
    cols = interactions[ITEM_COL].map(item_to_index)
    keep = rows.notna() & cols.notna()

    matrix = csr_matrix(
        (
            interactions.loc[keep, value_col].to_numpy(dtype=float),
            (rows[keep].to_numpy(dtype=int), cols[keep].to_numpy(dtype=int)),
        ),
        shape=(mapping.n_users, mapping.n_items),
    )
    matrix.sum_duplicates()
    return matrix
