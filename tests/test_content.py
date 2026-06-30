"""Tests for content-based filtering (tag profiles)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.config import ITEM_COL, USER_COL, WEIGHT_COL
from recsys.models.content import ContentBasedRecommender, build_artist_tag_profiles


def test_artist_profiles_shape_and_normalised(sample_tagged_artists):
    profiles, artist_ids = build_artist_tag_profiles(sample_tagged_artists)
    # 5 artists x 5 tags in the fixture.
    assert profiles.shape == (5, 5)
    assert list(artist_ids) == [1, 2, 3, 4, 5]
    # TF-IDF rows are L2-normalised -> each row norm is 1.
    norms = np.sqrt(profiles.multiply(profiles).sum(axis=1)).A.ravel()
    assert np.allclose(norms, 1.0)


def test_content_recommends_similar_tag_artists(sample_tagged_artists):
    # A user who only listens to artist 1 (electronic + idm) should be recommended the
    # other electronic+idm artists (4, 5) ahead of the rock artist (3).
    interactions = pd.DataFrame({USER_COL: [10], ITEM_COL: [1], WEIGHT_COL: [100]})
    model = ContentBasedRecommender(sample_tagged_artists).fit(interactions)
    recs = model.recommend(user_id=10, n=2)
    assert set(recs) == {4, 5}
    assert 3 not in recs


def test_content_excludes_seen(sample_tagged_artists):
    interactions = pd.DataFrame(
        {USER_COL: [10, 10], ITEM_COL: [1, 4], WEIGHT_COL: [100, 50]}
    )
    model = ContentBasedRecommender(sample_tagged_artists).fit(interactions)
    recs = model.recommend(user_id=10, n=5, exclude_seen=True)
    assert 1 not in recs and 4 not in recs


def test_raw_profiles_are_normalised_and_differ_from_tfidf(sample_tagged_artists):
    tfidf, _ = build_artist_tag_profiles(sample_tagged_artists, use_tfidf=True)
    raw, _ = build_artist_tag_profiles(sample_tagged_artists, use_tfidf=False)
    assert tfidf.shape == raw.shape
    # Raw rows are still L2-normalised.
    raw_norms = np.sqrt(raw.multiply(raw).sum(axis=1)).A.ravel()
    assert np.allclose(raw_norms, 1.0)
    # TF-IDF reweights, so the matrices are not identical.
    assert not np.allclose(tfidf.toarray(), raw.toarray())


def test_content_raw_variant_still_recommends(sample_tagged_artists):
    interactions = pd.DataFrame({USER_COL: [10], ITEM_COL: [1], WEIGHT_COL: [100]})
    model = ContentBasedRecommender(sample_tagged_artists, use_tfidf=False).fit(
        interactions
    )
    recs = model.recommend(user_id=10, n=2)
    assert set(recs) == {4, 5}  # electronic+idm neighbours, same as TF-IDF here


def test_content_unknown_user_returns_empty(sample_tagged_artists):
    interactions = pd.DataFrame({USER_COL: [10], ITEM_COL: [1], WEIGHT_COL: [100]})
    model = ContentBasedRecommender(sample_tagged_artists).fit(interactions)
    assert model.recommend(user_id=42, n=5) == []
