"""Social recommender: recommend what your friends listen to.

The Last.fm dataset ships a user friendship graph that the other methods ignore. This model
uses it directly: a user's score for an artist is the (log-scaled) amount their friends play
that artist. Structurally it is user-user collaborative filtering with the neighbour set
fixed to *declared friends* instead of taste-similar users - a useful contrast, since social
ties and taste similarity are not the same thing.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from recsys.config import USER_COL, WEIGHT_COL
from recsys.data.preprocess import (
    IndexMapping,
    build_index_mapping,
    build_interaction_matrix,
    log_scale,
)
from recsys.models.base import BaseRecommender
from recsys.models.cf import _top_n_item_ids


class SocialRecommender(BaseRecommender):
    """Recommend artists played by a user's friends."""

    def __init__(self, friends) -> None:
        self._friends = friends  # DataFrame [user_id, friend_id]
        self._mapping: IndexMapping | None = None

    def fit(self, interactions) -> "SocialRecommender":
        self._mapping = build_index_mapping(interactions)
        weighted = interactions.assign(_log=log_scale(interactions[WEIGHT_COL]))
        self._user_item = build_interaction_matrix(
            weighted, self._mapping, value_col="_log"
        )

        # Friendship adjacency (users x users); keep only edges between known users.
        user_to_index = {u: i for i, u in enumerate(self._mapping.user_ids)}
        rows = self._friends[USER_COL].map(user_to_index)
        cols = self._friends["friend_id"].map(user_to_index)
        keep = rows.notna() & cols.notna()
        n = self._mapping.n_users
        self._friend_adj = csr_matrix(
            (
                np.ones(int(keep.sum())),
                (rows[keep].to_numpy(int), cols[keep].to_numpy(int)),
            ),
            shape=(n, n),
        )
        return self

    def recommend(
        self, user_id: int, n: int = 10, exclude_seen: bool = True
    ) -> list[int]:
        if self._mapping is None:
            raise RuntimeError("call fit() before recommend()")
        try:
            uidx = self._mapping.user_index(user_id)
        except KeyError:
            return []
        # Sum of friends' (log) listening vectors.
        scores = np.asarray((self._friend_adj[uidx] @ self._user_item).todense()).ravel()
        seen = self._user_item.indices[
            self._user_item.indptr[uidx] : self._user_item.indptr[uidx + 1]
        ]
        return _top_n_item_ids(scores, self._mapping, n, seen, exclude_seen)
