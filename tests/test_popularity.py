"""Tests for the popularity baseline. Spec for the scoring you implement."""

from __future__ import annotations

from recsys.models.popularity import PopularityRecommender


def test_plays_strategy_ranks_by_total_weight(sample_interactions):
    # Plays totals: artist1=240, artist2=160, artist4=45, artist3=40, artist5=5
    model = PopularityRecommender(strategy="plays").fit(sample_interactions)
    assert model._ranked_items == [1, 2, 4, 3, 5]


def test_recommend_excludes_seen_items(sample_interactions):
    model = PopularityRecommender(strategy="plays").fit(sample_interactions)
    # User 1 listened to artists {1, 2, 3}; top remaining by plays are [4, 5].
    assert model.recommend(user_id=1, n=2, exclude_seen=True) == [4, 5]


def test_recommend_can_include_seen_items(sample_interactions):
    model = PopularityRecommender(strategy="plays").fit(sample_interactions)
    assert model.recommend(user_id=1, n=3, exclude_seen=False) == [1, 2, 4]


def test_recommend_respects_n(sample_interactions):
    model = PopularityRecommender(strategy="plays").fit(sample_interactions)
    assert len(model.recommend(user_id=4, n=2)) == 2
