"""Tests for implicit ALS matrix factorisation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from recsys.config import ITEM_COL, USER_COL, WEIGHT_COL
from recsys.models.mf import ImplicitALS, solve_factors


def test_solve_factors_matches_closed_form():
    # One entity observing two others with confidences [2, 3]; identity other-factors.
    confidence = csr_matrix(np.array([[2.0, 3.0]]))
    other = np.eye(2)
    reg = 0.1
    x = solve_factors(confidence, other, reg)
    # A = I + diag(c-1) + reg I = diag([2.1, 3.1]); b = c = [2, 3]; x = b / diag(A).
    assert x[0] == pytest.approx([2 / 2.1, 3 / 3.1])


def test_solve_factors_zero_row_stays_zero():
    confidence = csr_matrix((1, 3))  # empty row, no observations
    x = solve_factors(confidence, np.eye(3), regularization=0.1)
    assert np.allclose(x[0], 0.0)


def _clustered() -> pd.DataFrame:
    # Items 0,1,2 form cluster A (co-listened); items 3,4,5 form cluster B.
    rows = [
        (1, 0, 5),
        (1, 1, 5),
        (2, 0, 5),
        (2, 1, 5),
        (2, 2, 5),
        (3, 3, 5),
        (3, 4, 5),
        (4, 3, 5),
        (4, 4, 5),
        (4, 5, 5),
    ]
    return pd.DataFrame(rows, columns=[USER_COL, ITEM_COL, WEIGHT_COL])


def test_als_learns_cluster_structure():
    # User 1 listens to cluster-A items 0,1; item 2 (also cluster A, via user 2) should be
    # the top unseen recommendation, ahead of cluster-B items 3/4/5.
    model = ImplicitALS(factors=8, iterations=30, regularization=0.01, seed=0)
    model.fit(_clustered())
    recs = model.recommend(user_id=1, n=1)
    assert recs == [2]


def test_als_recommend_excludes_seen_and_respects_n(sample_interactions):
    model = ImplicitALS(factors=8, iterations=10, seed=0).fit(sample_interactions)
    recs = model.recommend(user_id=1, n=3, exclude_seen=True)
    assert all(r not in {1, 2, 3} for r in recs)  # user 1's seen artists
    assert len(recs) <= 3
    assert all(isinstance(r, int) for r in recs)


def test_als_unknown_user_returns_empty(sample_interactions):
    model = ImplicitALS(factors=4, iterations=5, seed=0).fit(sample_interactions)
    assert model.recommend(user_id=99999, n=5) == []
