"""Application service layer: load data, fit the active model, serve recommendations.

Built once at app startup and reused across requests. Keeping this out of ``app.py``
keeps the Flask routes thin and lets the same logic be reused from notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass

from recsys.config import (
    ARTISTS_FILE,
    DEFAULT_SEED,
    DEFAULT_TOP_N,
    ITEM_COL,
    RAW_DIR,
    USER_COL,
    USER_ARTISTS_FILE,
)
from recsys.data.loader import load_artists, load_user_artists
from recsys.data.split import leave_n_out_split
from recsys.eval.harness import evaluate
from recsys.models.popularity import PopularityRecommender


@dataclass
class Recommendation:
    """A single recommended artist for display."""

    artist_id: int
    name: str


class RecommenderService:
    """Loads the dataset, fits the popularity baseline, and serves recommendations."""

    def __init__(self, top_n: int = DEFAULT_TOP_N) -> None:
        self.top_n = top_n
        interactions = load_user_artists(RAW_DIR / USER_ARTISTS_FILE)
        artists = load_artists(RAW_DIR / ARTISTS_FILE)
        self._artist_name = dict(zip(artists[ITEM_COL], artists["name"]))

        self._train, self._test = leave_n_out_split(interactions, seed=DEFAULT_SEED)
        self._model = PopularityRecommender(strategy="plays").fit(self._train)
        self.user_ids = sorted(self._train[USER_COL].unique().tolist())

        # Offline quality of the active model on the held-out split (the 30% story).
        self._metrics = evaluate(
            PopularityRecommender(strategy="plays"),
            self._train,
            self._test,
            k=self.top_n,
        )

    def recommend(self, user_id: int) -> list[Recommendation]:
        """Top-N recommendations for a user, as display-ready records."""
        ids = self._model.recommend(user_id, n=self.top_n, exclude_seen=True)
        return [
            Recommendation(artist_id=aid, name=self._artist_name.get(aid, f"#{aid}"))
            for aid in ids
        ]

    @property
    def precision_at_k(self) -> float:
        return self._metrics["precision_at_k"]

    @property
    def k(self) -> int:
        return int(self._metrics["k"])

    @property
    def n_eval_users(self) -> int:
        return int(self._metrics["n_users"])
