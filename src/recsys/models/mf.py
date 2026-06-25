"""Matrix factorisation for implicit feedback: Alternating Least Squares (ALS).

Hand-implementation of Hu, Koren & Volinsky (2008), "Collaborative Filtering for Implicit
Feedback Datasets". The idea:

- Treat every observed play as a *preference* of 1 with a *confidence* that grows with the
  play count: ``c_ui = 1 + alpha * log(1 + plays)``. Unobserved pairs have preference 0 and
  the baseline confidence 1 (a weak "probably not").
- Learn user factors X (users x f) and item factors Y (items x f) so that ``x_u . y_i``
  approximates the preference, weighted by confidence.
- ALS alternates: fix Y and solve each user's factors in closed form, then fix X and solve
  each item's, repeating. Each half-step is a ridge regression, which is stable and fast.

The per-entity update uses the Hu et al. trick ``Yt Cu Y = YtY + Yt (Cu - I) Y`` so the
shared ``YtY`` is computed once per iteration rather than per user.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from recsys.config import DEFAULT_SEED, ITEM_COL, USER_COL, WEIGHT_COL
from recsys.data.preprocess import IndexMapping, build_index_mapping
from recsys.models.base import BaseRecommender
from recsys.models.cf import _top_n_item_ids


def solve_factors(
    confidence: csr_matrix,
    other_factors: np.ndarray,
    regularization: float,
) -> np.ndarray:
    """One ALS half-step: solve every row's factors given the fixed other factors.

    For entity e (e.g. a user) with observed counterparts i and confidences ``c_ei``:

        A = YtY + Y_obs^T diag(c_ei - 1) Y_obs + reg * I
        b = Y_obs^T c_ei                       (preference is 1 on observed entries)
        x_e = A^-1 b

    Parameters
    ----------
    confidence : scipy.sparse.csr_matrix
        Sparse ``(n_entities, n_others)`` matrix whose stored values are the confidences
        ``c_ei`` for observed pairs.
    other_factors : np.ndarray
        The fixed factor matrix ``(n_others, f)``.
    regularization : float
        Ridge term added to the diagonal.

    Returns
    -------
    np.ndarray
        Updated factor matrix ``(n_entities, f)``.
    """
    n_entities = confidence.shape[0]
    f = other_factors.shape[1]
    YtY = other_factors.T @ other_factors
    reg_eye = regularization * np.eye(f)
    factors = np.zeros((n_entities, f))

    for e in range(n_entities):
        start, end = confidence.indptr[e], confidence.indptr[e + 1]
        idx = confidence.indices[start:end]
        if idx.size == 0:
            continue
        c = confidence.data[start:end]  # confidences c_ei for observed items
        Y_obs = other_factors[idx]  # (m, f)
        # A = YtY + Y_obs^T diag(c - 1) Y_obs + reg I
        A = YtY + (Y_obs * (c - 1.0)[:, None]).T @ Y_obs + reg_eye
        # b = Y_obs^T c   (preference 1 on observed entries)
        b = Y_obs.T @ c
        factors[e] = np.linalg.solve(A, b)
    return factors


class ImplicitALS(BaseRecommender):
    """Implicit-feedback matrix factorisation via Alternating Least Squares."""

    def __init__(
        self,
        factors: int = 64,
        regularization: float = 0.05,
        alpha: float = 40.0,
        iterations: int = 15,
        seed: int = DEFAULT_SEED,
    ) -> None:
        self.factors = factors
        self.regularization = regularization
        self.alpha = alpha
        self.iterations = iterations
        self.seed = seed
        self._mapping: IndexMapping | None = None

    def _build_confidence(self, interactions) -> csr_matrix:
        """Confidence matrix c_ui = 1 + alpha * log(1 + plays) for observed pairs."""
        mapping = self._mapping
        user_to_index = {u: i for i, u in enumerate(mapping.user_ids)}
        item_to_index = {a: i for i, a in enumerate(mapping.item_ids)}
        rows = interactions[USER_COL].map(user_to_index)
        cols = interactions[ITEM_COL].map(item_to_index)
        conf = 1.0 + self.alpha * np.log1p(interactions[WEIGHT_COL].to_numpy(float))
        return csr_matrix(
            (conf, (rows.to_numpy(int), cols.to_numpy(int))),
            shape=(mapping.n_users, mapping.n_items),
        )

    def fit(self, interactions) -> "ImplicitALS":
        self._mapping = build_index_mapping(interactions)
        cui = self._build_confidence(interactions)
        ciu = cui.T.tocsr()

        rng = np.random.default_rng(self.seed)
        self._user_factors = 0.01 * rng.standard_normal(
            (self._mapping.n_users, self.factors)
        )
        self._item_factors = 0.01 * rng.standard_normal(
            (self._mapping.n_items, self.factors)
        )
        self._user_item = cui  # confidences double as the "seen" mask

        for _ in range(self.iterations):
            self._user_factors = solve_factors(
                cui, self._item_factors, self.regularization
            )
            self._item_factors = solve_factors(
                ciu, self._user_factors, self.regularization
            )
        return self

    def recommend(
        self, user_id: int, n: int = 10, exclude_seen: bool = True
    ) -> list[int]:
        if self._mapping is None:
            raise RuntimeError("call fit() before recommend()")
        try:
            uidx = self._mapping.user_index(user_id)
        except KeyError:
            return []
        scores = self._item_factors @ self._user_factors[uidx]
        seen = self._user_item.indices[
            self._user_item.indptr[uidx] : self._user_item.indptr[uidx + 1]
        ]
        return _top_n_item_ids(
            scores, self._mapping, n, seen, exclude_seen, positive_only=False
        )
