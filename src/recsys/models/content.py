"""Content-based filtering using artist tags.

Where collaborative filtering only knows *who co-listens*, content-based filtering knows
*what an artist is* (its tags: genres, moods, eras). The recipe:

1. Build an artist x tag matrix of how often each tag was applied to each artist.
2. Weight it with TF-IDF so distinctive tags ("witch house") count for more than ubiquitous
   ones ("rock"). Each artist becomes a unit-length vector in "tag space".
3. A user's profile is the play-weighted average of the tag vectors of artists they listen
   to - a point in the same tag space.
4. Recommend the artists whose tag vector is most cosine-similar to the user's profile.

A side benefit: because it only needs an artist's tags (not its listening history), it can
recommend cold-start artists that collaborative filtering never could.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

from recsys.config import ITEM_COL, USER_COL, WEIGHT_COL
from recsys.data.preprocess import (
    IndexMapping,
    build_interaction_matrix,
    log_scale,
)
from recsys.models.base import BaseRecommender
from recsys.models.cf import _top_n_item_ids


def build_artist_tag_profiles(
    tagged_artists: pd.DataFrame,
) -> tuple[csr_matrix, np.ndarray]:
    """Build L2-normalised TF-IDF artist profiles from tag assignments.

    Parameters
    ----------
    tagged_artists : pandas.DataFrame
        Columns ``[user_id, artist_id, tag_id]`` (one row per assignment).

    Returns
    -------
    (profiles, artist_ids) : tuple
        ``profiles`` is a sparse ``(n_artists, n_tags)`` TF-IDF matrix (rows unit-norm);
        ``artist_ids`` gives the artist id for each row.
    """
    artist_ids = np.sort(tagged_artists[ITEM_COL].unique())
    tag_ids = np.sort(tagged_artists["tag_id"].unique())
    artist_to_row = {a: i for i, a in enumerate(artist_ids)}
    tag_to_col = {t: i for i, t in enumerate(tag_ids)}

    counts = tagged_artists.groupby([ITEM_COL, "tag_id"]).size().reset_index(name="n")
    rows = counts[ITEM_COL].map(artist_to_row).to_numpy()
    cols = counts["tag_id"].map(tag_to_col).to_numpy()
    count_matrix = csr_matrix(
        (counts["n"].to_numpy(dtype=float), (rows, cols)),
        shape=(len(artist_ids), len(tag_ids)),
    )

    profiles = TfidfTransformer(norm="l2").fit_transform(count_matrix)
    return profiles.tocsr(), artist_ids


class ContentBasedRecommender(BaseRecommender):
    """Recommend artists whose tag profile matches the user's listening-weighted taste."""

    def __init__(self, tagged_artists: pd.DataFrame) -> None:
        self._tagged_artists = tagged_artists
        self._artist_profiles: csr_matrix | None = None
        self._mapping: IndexMapping | None = None

    def fit(self, interactions: pd.DataFrame) -> "ContentBasedRecommender":
        # Artist tag space (independent of the train split).
        self._artist_profiles, artist_ids = build_artist_tag_profiles(
            self._tagged_artists
        )
        user_ids = np.sort(interactions[USER_COL].unique())
        # Item axis is the *tagged* artists; interactions on untagged artists are dropped.
        self._mapping = IndexMapping(user_ids=user_ids, item_ids=artist_ids)

        weighted = interactions.assign(_log=log_scale(interactions[WEIGHT_COL]))
        self._user_item = build_interaction_matrix(
            weighted, self._mapping, value_col="_log"
        )
        # User profile = play-weighted sum of artist tag vectors, then unit-normalised.
        user_profiles = self._user_item @ self._artist_profiles
        self._user_profiles = normalize(user_profiles, norm="l2", axis=1).tocsr()
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
        profile = self._user_profiles[uidx]
        scores = np.asarray((profile @ self._artist_profiles.T).todense()).ravel()
        seen = self._user_item.indices[
            self._user_item.indptr[uidx] : self._user_item.indptr[uidx + 1]
        ]
        return _top_n_item_ids(scores, self._mapping, n, seen, exclude_seen)
