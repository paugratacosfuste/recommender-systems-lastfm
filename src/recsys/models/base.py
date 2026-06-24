"""The recommender interface every model implements.

A single, stable contract (``fit`` then ``recommend``) lets the evaluation harness and
the Flask app treat all methods interchangeably - which is exactly what the assignment's
"compare methods on the same data" goal requires.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseRecommender(ABC):
    """Abstract base class for all recommenders."""

    @abstractmethod
    def fit(self, interactions: pd.DataFrame) -> "BaseRecommender":
        """Train the model on a set of interactions.

        Parameters
        ----------
        interactions : pandas.DataFrame
            Training interactions with ``[user_id, artist_id, weight]`` columns.

        Returns
        -------
        BaseRecommender
            ``self``, to allow chaining.
        """

    @abstractmethod
    def recommend(
        self, user_id: int, n: int = 10, exclude_seen: bool = True
    ) -> list[int]:
        """Return the top-``n`` recommended artist ids for ``user_id``.

        Parameters
        ----------
        user_id : int
            The user to recommend for.
        n : int
            Number of recommendations to return.
        exclude_seen : bool
            If True, omit artists the user already interacted with in training.

        Returns
        -------
        list of int
            Recommended artist ids, ordered most- to least-relevant.
        """
