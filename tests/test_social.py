"""Tests for the friendship-based social recommender."""

from __future__ import annotations

from recsys.models.social import SocialRecommender


def test_social_recommends_from_friends(sample_interactions, sample_friends):
    # User 1's friends are {2, 3}. Friend 2 plays artists {1,2,4}, friend 3 plays {1,3,5}.
    # Excluding user 1's seen {1,2,3}, the friend-listened unseen artists are 4 and 5,
    # with 4 (20 plays) ranked above 5 (5 plays).
    model = SocialRecommender(sample_friends).fit(sample_interactions)
    assert model.recommend(user_id=1, n=5) == [4, 5]


def test_social_excludes_seen(sample_interactions, sample_friends):
    model = SocialRecommender(sample_friends).fit(sample_interactions)
    recs = model.recommend(user_id=1, n=5, exclude_seen=True)
    assert all(r not in {1, 2, 3} for r in recs)


def test_social_unknown_user_returns_empty(sample_interactions, sample_friends):
    model = SocialRecommender(sample_friends).fit(sample_interactions)
    assert model.recommend(user_id=99999, n=5) == []
