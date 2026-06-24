"""Tests for the data loaders."""

from __future__ import annotations

import pytest

from recsys.config import ITEM_COL, SAMPLE_DIR, USER_COL, WEIGHT_COL
from recsys.data.loader import load_artists, load_user_artists


def test_load_user_artists_standardises_columns(sample_interactions):
    assert list(sample_interactions.columns) == [USER_COL, ITEM_COL, WEIGHT_COL]
    assert len(sample_interactions) == 11
    assert (sample_interactions[WEIGHT_COL] > 0).all()


def test_load_user_artists_weights_match_fixture(sample_interactions):
    plays = sample_interactions.groupby(ITEM_COL)[WEIGHT_COL].sum().to_dict()
    assert plays == {1: 240, 2: 160, 3: 40, 4: 45, 5: 5}


def test_load_artists_returns_id_and_name():
    artists = load_artists(SAMPLE_DIR / "artists.dat")
    assert list(artists.columns) == [ITEM_COL, "name"]
    assert artists.loc[artists[ITEM_COL] == 1, "name"].iloc[0] == "Aphex Twin"


def test_missing_file_raises_helpful_error():
    with pytest.raises(FileNotFoundError, match="download_data"):
        load_user_artists(SAMPLE_DIR / "does_not_exist.dat")
