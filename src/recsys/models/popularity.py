"""Non-personalised popularity baseline.

This is the comparison floor for every later method. It recommends the globally most
popular artists, optionally excluding ones the user already listened to. The *definition*
of "popular" is a real design choice with bias implications, left to you below.
"""

from __future__ import annotations

import pandas as pd

from recsys.config import ITEM_COL, USER_COL, WEIGHT_COL
from recsys.models.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """Recommend the most popular artists to everyone.

    Parameters
    ----------
    strategy : str
        How to measure popularity. One of:
        - ``"plays"``     : total listening weight summed across users.
        - ``"listeners"`` : number of distinct users who listened to the artist.
        - ``"damped"``    : total plays damped by a constant to curb mega-popular artists.
    damping : float
        Constant used only by the ``"damped"`` strategy (see your implementation).
    """

    def __init__(self, strategy: str = "plays", damping: float = 100.0) -> None:
        self.strategy = strategy
        self.damping = damping
        self._ranked_items: list[int] = []
        self._seen: dict[int, set[int]] = {}

    def fit(self, interactions: pd.DataFrame) -> "PopularityRecommender":
        """Compute the global popularity ranking and per-user seen sets."""
        self._seen = interactions.groupby(USER_COL)[ITEM_COL].apply(set).to_dict()
        self._ranked_items = self._rank_items(interactions)
        return self

    def _rank_items(self, interactions: pd.DataFrame) -> list[int]:
        """Rank all artists from most to least popular per ``self.strategy``.

        This scoring choice shapes popularity bias in every downstream comparison:
          - "plays":     total listening weight (a few super-fans can dominate).
          - "listeners": number of distinct users (rewards broad appeal).
          - "damped":    total plays / (n_listeners + damping), curbing mega-hits.
        """
        grouped = interactions.groupby(ITEM_COL)
        if self.strategy == "plays":
            scores = grouped[WEIGHT_COL].sum()
        elif self.strategy == "listeners":
            scores = grouped[USER_COL].nunique()
        elif self.strategy == "damped":
            scores = grouped[WEIGHT_COL].sum() / (
                grouped[USER_COL].nunique() + self.damping
            )
        else:
            raise ValueError(
                f"Unknown strategy {self.strategy!r}; "
                "expected 'plays', 'listeners', or 'damped'."
            )
        return scores.sort_values(ascending=False).index.tolist()

    def recommend(
        self, user_id: int, n: int = 10, exclude_seen: bool = True
    ) -> list[int]:
        """Return the top-``n`` popular artists not already seen by ``user_id``."""
        seen = self._seen.get(user_id, set()) if exclude_seen else set()
        recommendations = [item for item in self._ranked_items if item not in seen]
        return recommendations[:n]
