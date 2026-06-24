"""Tests for the evaluation harness (fit -> score over test users)."""

from __future__ import annotations

import pandas as pd

from recsys.config import ITEM_COL, USER_COL, WEIGHT_COL
from recsys.eval.harness import evaluate
from recsys.models.popularity import PopularityRecommender


def _frame(rows: list[tuple[int, int, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[USER_COL, ITEM_COL, WEIGHT_COL])


def test_evaluate_averages_precision_over_users():
    # Train popularity order by plays: artist1=18, artist2=5, artist3=4 -> [1, 2, 3].
    train = _frame([(1, 1, 10), (1, 2, 5), (2, 1, 8), (2, 3, 4)])
    # Each user's single held-out artist is the only one left after excluding seen,
    # so a correct top-1 recommendation scores precision@1 = 1.0 for both users.
    test = _frame([(1, 3, 1), (2, 2, 1)])

    result = evaluate(PopularityRecommender(strategy="plays"), train, test, k=1)

    assert result["precision_at_k"] == 1.0
    assert result["n_users"] == 2.0
    assert result["k"] == 1.0


def test_evaluate_handles_a_miss():
    train = _frame([(1, 1, 10), (1, 2, 5), (2, 1, 8), (2, 2, 4)])
    # User 1 has seen {1, 2}; popularity recommends nothing relevant for a held-out
    # artist 9 that is not even in the catalogue -> precision 0.
    test = _frame([(1, 9, 1)])

    result = evaluate(PopularityRecommender(strategy="plays"), train, test, k=3)

    assert result["precision_at_k"] == 0.0
    assert result["n_users"] == 1.0
