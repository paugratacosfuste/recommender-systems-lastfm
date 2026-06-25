"""Memory-based collaborative filtering (k-nearest-neighbours).

Two flavours, both built on cosine similarity over the implicit, log-scaled interaction
matrix:

- :class:`ItemKNNRecommender` - "artists similar to the ones you already play".
  Similarity is computed between artists (columns), which is stable and the standard
  choice for implicit feedback.
- :class:`UserKNNRecommender` - "artists played by users with taste similar to yours".

The maths is hand-implemented (cosine = normalise rows, then dot product); scipy sparse
operations and sklearn's ``normalize`` provide the speed.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

from recsys.config import WEIGHT_COL
from recsys.data.preprocess import (
    IndexMapping,
    build_index_mapping,
    build_interaction_matrix,
    log_scale,
)
from recsys.models.base import BaseRecommender


def cosine_similarity_rows(
    matrix: csr_matrix, k_neighbors: int | None = None
) -> csr_matrix:
    """Row-wise cosine similarity of a sparse matrix, with a zeroed diagonal.

    cos(a, b) = (a . b) / (||a|| ||b||). Normalising each row to unit length first turns
    the cosine into a plain dot product, so the whole similarity matrix is one sparse
    product ``normalised @ normalised.T``.

    Parameters
    ----------
    matrix : scipy.sparse.csr_matrix
        Rows are the entities to compare (items for item-item, users for user-user).
    k_neighbors : int, optional
        If given, keep only each row's top-``k`` most similar neighbours (bounds memory
        and denoises). If None, keep the full similarity matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        Square similarity matrix; entry (i, j) is the cosine similarity of rows i and j,
        with the self-similarity diagonal set to zero.
    """
    normalised = normalize(matrix, norm="l2", axis=1)
    similarity = (normalised @ normalised.T).tocsr()
    similarity.setdiag(0.0)
    similarity.eliminate_zeros()
    if k_neighbors is not None:
        similarity = _keep_top_k_per_row(similarity, k_neighbors)
    return similarity


def _keep_top_k_per_row(matrix: csr_matrix, k: int) -> csr_matrix:
    """Keep only the ``k`` largest values in each row of a CSR matrix."""
    matrix = matrix.tocsr()
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row], matrix.indptr[row + 1]
        if end - start > k:
            row_data = matrix.data[start:end]
            # Indices of the (end-start-k) smallest values, which we zero out.
            drop = np.argpartition(row_data, end - start - k)[: end - start - k]
            matrix.data[start + drop] = 0.0
    matrix.eliminate_zeros()
    return matrix


def _build_user_item(interactions, mapping: IndexMapping) -> csr_matrix:
    """User-item matrix of log-scaled weights (dampens superfan play counts)."""
    weighted = interactions.assign(_log=log_scale(interactions[WEIGHT_COL]))
    return build_interaction_matrix(weighted, mapping, value_col="_log")


def _top_n_item_ids(
    scores: np.ndarray,
    mapping: IndexMapping,
    n: int,
    seen_idx: np.ndarray,
    exclude_seen: bool,
    positive_only: bool = True,
) -> list[int]:
    """Pick the n highest-scoring unseen items and map them to artist ids.

    With ``positive_only`` (default), items with non-positive scores are dropped - right
    for similarity/count scores where 0 means "no signal". Matrix factorisation scores can
    legitimately be negative, so it passes ``positive_only=False`` to keep the top-n.
    """
    scores = scores.copy()
    if exclude_seen and seen_idx.size:
        scores[seen_idx] = -np.inf
    # Partial sort for the top candidates, then order them descending.
    n_candidates = min(n, scores.size)
    top = np.argpartition(scores, -n_candidates)[-n_candidates:]
    top = top[np.argsort(scores[top])[::-1]]
    threshold = -np.inf if not positive_only else 0.0
    return [int(mapping.item_ids[i]) for i in top if scores[i] > threshold]


class _MatrixCFRecommender(BaseRecommender):
    """Shared plumbing for matrix-based CF: build mapping, matrix, and seen sets."""

    def __init__(self, k_neighbors: int | None = None) -> None:
        self.k_neighbors = k_neighbors
        self._mapping: IndexMapping | None = None
        self._user_item: csr_matrix | None = None

    def _prepare(self, interactions) -> None:
        self._mapping = build_index_mapping(interactions)
        self._user_item = _build_user_item(interactions, self._mapping)

    def _seen_indices(self, user_index: int) -> np.ndarray:
        return self._user_item.indices[
            self._user_item.indptr[user_index] : self._user_item.indptr[user_index + 1]
        ]


class ItemKNNRecommender(_MatrixCFRecommender):
    """Item-item kNN: score an artist by its similarity to the user's played artists."""

    def fit(self, interactions) -> "ItemKNNRecommender":
        self._prepare(interactions)
        # Similarity between items = cosine over the item-user matrix (matrix transposed).
        self._item_sim = cosine_similarity_rows(
            self._user_item.T.tocsr(), k_neighbors=self.k_neighbors
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
        user_row = self._user_item[uidx]
        scores = np.asarray((user_row @ self._item_sim).todense()).ravel()
        return _top_n_item_ids(
            scores, self._mapping, n, self._seen_indices(uidx), exclude_seen
        )


class UserKNNRecommender(_MatrixCFRecommender):
    """User-user kNN: score an artist by how much similar users played it."""

    def fit(self, interactions) -> "UserKNNRecommender":
        self._prepare(interactions)
        self._user_sim = cosine_similarity_rows(
            self._user_item, k_neighbors=self.k_neighbors
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
        sim_row = self._user_sim[uidx]
        scores = np.asarray((sim_row @ self._user_item).todense()).ravel()
        return _top_n_item_ids(
            scores, self._mapping, n, self._seen_indices(uidx), exclude_seen
        )
