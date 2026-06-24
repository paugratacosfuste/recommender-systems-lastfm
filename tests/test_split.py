"""Tests for the per-user train/test split."""

from __future__ import annotations

import pytest

from recsys.config import ITEM_COL, USER_COL
from recsys.data.split import leave_n_out_split


def test_split_is_disjoint_and_complete(sample_interactions):
    train, test = leave_n_out_split(sample_interactions, test_frac=0.34, seed=0)
    # No interaction appears in both sets, and together they reconstruct the input.
    combined = train.merge(test, how="inner", on=[USER_COL, ITEM_COL]).shape[0]
    assert combined == 0
    assert len(train) + len(test) == len(sample_interactions)


def test_every_test_user_is_present_in_train(sample_interactions):
    train, test = leave_n_out_split(sample_interactions, test_frac=0.34, seed=0)
    assert set(test[USER_COL]).issubset(set(train[USER_COL]))


def test_min_train_is_respected(sample_interactions):
    train, _ = leave_n_out_split(sample_interactions, test_frac=0.9, min_train=1, seed=0)
    counts = train.groupby(USER_COL)[ITEM_COL].size()
    assert (counts >= 1).all()


def test_split_is_reproducible(sample_interactions):
    a_train, a_test = leave_n_out_split(sample_interactions, seed=7)
    b_train, b_test = leave_n_out_split(sample_interactions, seed=7)
    assert a_test.equals(b_test)
    assert a_train.equals(b_train)


def test_invalid_test_frac_raises(sample_interactions):
    with pytest.raises(ValueError):
        leave_n_out_split(sample_interactions, test_frac=1.5)
