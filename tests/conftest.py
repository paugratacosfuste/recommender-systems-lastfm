"""Shared test fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from recsys.config import SAMPLE_DIR
from recsys.data.loader import (
    load_tags,
    load_user_artists,
    load_user_friends,
    load_user_tagged_artists,
)


@pytest.fixture
def sample_interactions() -> pd.DataFrame:
    """The committed tiny fixture as a standardised interactions DataFrame.

    Plays totals (for reference in tests):
        artist1=240, artist2=160, artist4=45, artist3=40, artist5=5
    """
    return load_user_artists(SAMPLE_DIR / "user_artists.dat")


@pytest.fixture
def sample_tagged_artists() -> pd.DataFrame:
    """Tiny tag fixture: artists 1/4/5 = electronic+idm, 2 = electronic+art pop,
    3 = alternative+rock."""
    return load_user_tagged_artists(SAMPLE_DIR / "user_taggedartists.dat")


@pytest.fixture
def sample_friends() -> pd.DataFrame:
    """Tiny social graph: 1<->2, 1<->3, 3<->4."""
    return load_user_friends(SAMPLE_DIR / "user_friends.dat")


@pytest.fixture
def sample_tags() -> pd.DataFrame:
    """Tag vocabulary fixture: 1=electronic, 2=idm, 3=art pop, 4=alternative, 5=rock."""
    return load_tags(SAMPLE_DIR / "tags.dat")
