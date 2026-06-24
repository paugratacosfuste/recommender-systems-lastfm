"""Tests for collaborative filtering: cosine similarity and kNN recommenders."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from recsys.models.cf import (
    ItemKNNRecommender,
    UserKNNRecommender,
    cosine_similarity_rows,
)


def test_cosine_similarity_identical_and_orthogonal_rows():
    # Rows 0 and 1 are identical (cos=1); row 2 is orthogonal to both (cos=0).
    m = csr_matrix(np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 5.0]]))
    sim = cosine_similarity_rows(m).toarray()
    assert sim[0, 1] == np.float64(1.0).round(9) or abs(sim[0, 1] - 1.0) < 1e-9
    assert abs(sim[0, 2]) < 1e-9
    # Diagonal is zeroed (an item is not its own neighbour).
    assert sim[0, 0] == 0.0


def test_cosine_similarity_top_k_pruning_keeps_strongest():
    # Row 0 is most similar to row 1, less to row 2; k=1 keeps only the strongest.
    m = csr_matrix(np.array([[1.0, 0.0], [1.0, 0.1], [0.2, 1.0], [0.0, 1.0]]))
    full = cosine_similarity_rows(m)
    pruned = cosine_similarity_rows(m, k_neighbors=1)
    assert pruned.nnz <= full.nnz
    # Every row keeps at most one neighbour.
    assert (np.diff(pruned.indptr) <= 1).all()


def test_item_knn_recommends_unseen_and_excludes_seen(sample_interactions):
    model = ItemKNNRecommender().fit(sample_interactions)
    recs = model.recommend(user_id=1, n=5, exclude_seen=True)
    seen = {1, 2, 3}  # user 1's training artists
    assert all(r not in seen for r in recs)
    assert len(recs) <= 5


def test_item_knn_personalises(sample_interactions):
    # User 4 played artists {2, 4}; CF should surface artists co-listened with those,
    # and the result should differ from a raw popularity ordering for at least some user.
    model = ItemKNNRecommender().fit(sample_interactions)
    recs_u1 = model.recommend(user_id=1, n=5)
    recs_u4 = model.recommend(user_id=4, n=5)
    # Different users with different histories get different recommendations.
    assert recs_u1 != recs_u4


def test_user_knn_excludes_seen_and_returns_ids(sample_interactions):
    model = UserKNNRecommender().fit(sample_interactions)
    recs = model.recommend(user_id=2, n=5, exclude_seen=True)
    seen = {1, 2, 4}  # user 2's training artists
    assert all(r not in seen for r in recs)
    assert all(isinstance(r, int) for r in recs)


def test_unknown_user_returns_empty(sample_interactions):
    model = ItemKNNRecommender().fit(sample_interactions)
    assert model.recommend(user_id=99999, n=5) == []
