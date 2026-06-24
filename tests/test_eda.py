"""Tests for exploratory statistics."""

from __future__ import annotations

import numpy as np
import pytest

from recsys.data.eda import gini, popularity_curve, summary_stats, top_k_play_share


def test_gini_of_equal_distribution_is_zero():
    assert gini(np.array([5, 5, 5, 5])) == pytest.approx(0.0, abs=1e-9)


def test_gini_of_concentrated_distribution_is_high():
    # One artist has everything -> approaches (n-1)/n = 0.8 for n=5.
    assert gini(np.array([0, 0, 0, 0, 100])) == pytest.approx(0.8, abs=1e-9)


def test_gini_rejects_negative_values():
    with pytest.raises(ValueError):
        gini(np.array([-1.0, 2.0]))


def test_gini_handles_empty_and_zero():
    assert gini(np.array([])) == 0.0
    assert gini(np.array([0, 0, 0])) == 0.0


def test_summary_stats_on_fixture(sample_interactions):
    stats = summary_stats(sample_interactions)
    assert stats["n_users"] == 4
    assert stats["n_items"] == 5
    assert stats["n_interactions"] == 11
    # 11 observed cells out of 4*5 = 20 -> density 0.55, sparsity 0.45.
    assert stats["density"] == pytest.approx(0.55)
    assert stats["sparsity"] == pytest.approx(0.45)


def test_popularity_curve_is_descending(sample_interactions):
    curve = popularity_curve(sample_interactions)
    assert list(curve) == sorted(curve, reverse=True)
    assert curve.iloc[0] == 240  # artist 1 is most played


def test_top_k_play_share(sample_interactions):
    # Top-1 artist (240 plays) over total plays (490) -> ~0.4898.
    assert top_k_play_share(sample_interactions, k=1) == pytest.approx(240 / 490)
    assert top_k_play_share(sample_interactions, k=5) == pytest.approx(1.0)
