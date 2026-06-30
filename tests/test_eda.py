"""Tests for exploratory statistics."""

from __future__ import annotations

import numpy as np
import pytest

from recsys.data.eda import (
    friend_degree,
    friend_listening_overlap,
    gini,
    interaction_quality,
    most_active_users,
    popularity_curve,
    summary_stats,
    tag_frequency,
    tags_per_artist,
    top_k_play_share,
)


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


def test_most_active_users(sample_interactions):
    from recsys.config import USER_COL

    top = most_active_users(sample_interactions, n=2)
    assert list(top.columns) == [USER_COL, "n_artists", "total_plays"]
    assert len(top) == 2
    # Users 1, 2, 3 have 3 artists each; user 4 has only 2, so never in the top 2.
    assert (top["n_artists"] == 3).all()
    assert 4 not in set(top[USER_COL])


def test_interaction_quality(sample_interactions):
    q = interaction_quality(sample_interactions)
    assert q["n_rows"] == 11
    assert q["n_duplicate_pairs"] == 0
    assert q["n_nonpositive_weight"] == 0
    assert q["max_weight"] == 100
    assert q["min_weight"] == 5
    assert q["median_weight"] == 40


def test_tag_frequency(sample_tagged_artists, sample_tags):
    top = tag_frequency(sample_tagged_artists, sample_tags, n=2)
    # Tag 1 (electronic) is applied to artists 1,2,4,5 -> 4 assignments (the most).
    assert top.iloc[0]["tag_id"] == 1
    assert top.iloc[0]["assignments"] == 4
    assert top.iloc[0]["tag"] == "electronic"


def test_tags_per_artist(sample_tagged_artists):
    tpa = tags_per_artist(sample_tagged_artists)
    # Every fixture artist has exactly two tags.
    assert (tpa == 2).all()
    assert len(tpa) == 5


def test_friend_degree(sample_friends):
    deg = friend_degree(sample_friends)
    assert deg[1] == 2  # user 1 is friends with 2 and 3
    assert deg[2] == 1
    assert deg[4] == 1


def test_friend_listening_overlap(sample_interactions, sample_friends):
    res = friend_listening_overlap(
        sample_interactions, sample_friends, n_random=50, seed=0
    )
    # Friend pairs (1,2)=0.5, (1,3)=0.5, (3,4)=0.0 -> mean 1/3.
    assert res["friend_jaccard"] == pytest.approx(1 / 3)
    assert 0.0 <= res["random_jaccard"] <= 1.0


def test_top_k_play_share(sample_interactions):
    # Top-1 artist (240 plays) over total plays (490) -> ~0.4898.
    assert top_k_play_share(sample_interactions, k=1) == pytest.approx(240 / 490)
    assert top_k_play_share(sample_interactions, k=5) == pytest.approx(1.0)
