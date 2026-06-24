"""Shared test fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from recsys.config import SAMPLE_DIR
from recsys.data.loader import load_user_artists


@pytest.fixture
def sample_interactions() -> pd.DataFrame:
    """The committed tiny fixture as a standardised interactions DataFrame.

    Plays totals (for reference in tests):
        artist1=240, artist2=160, artist4=45, artist3=40, artist5=5
    """
    return load_user_artists(SAMPLE_DIR / "user_artists.dat")
