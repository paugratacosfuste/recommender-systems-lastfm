"""Application service layer: load data, fit the available models, serve recommendations.

Built once at app startup and reused across requests. A small method registry maps each
selectable label to a fitted model plus its offline metrics (accuracy and beyond-accuracy).
Later modules plug into the same registry - the app code does not change.
"""

from __future__ import annotations

from dataclasses import dataclass

from recsys.config import (
    ARTISTS_FILE,
    DEFAULT_SEED,
    DEFAULT_TOP_N,
    ITEM_COL,
    RAW_DIR,
    USER_ARTISTS_FILE,
    USER_COL,
    USER_TAGGED_ARTISTS_FILE,
)
from recsys.data.loader import (
    load_artists,
    load_user_artists,
    load_user_tagged_artists,
)
from recsys.data.split import leave_n_out_split
from recsys.eval.beyond_accuracy import BeyondAccuracyInputs, build_item_popularity
from recsys.eval.harness import evaluate
from recsys.models.base import BaseRecommender
from recsys.models.cf import ItemKNNRecommender, UserKNNRecommender
from recsys.models.content import ContentBasedRecommender, build_artist_tag_profiles
from recsys.models.popularity import PopularityRecommender


@dataclass
class Recommendation:
    """A single recommended artist for display."""

    artist_id: int
    name: str


class RecommenderService:
    """Loads the dataset, fits every available method, and serves recommendations."""

    def __init__(self, top_n: int = DEFAULT_TOP_N) -> None:
        self.top_n = top_n
        interactions = load_user_artists(RAW_DIR / USER_ARTISTS_FILE)
        tagged = load_user_tagged_artists(RAW_DIR / USER_TAGGED_ARTISTS_FILE)
        artists = load_artists(RAW_DIR / ARTISTS_FILE)
        self._artist_name = dict(zip(artists[ITEM_COL], artists["name"]))

        self._train, self._test = leave_n_out_split(interactions, seed=DEFAULT_SEED)
        self.user_ids = sorted(self._train[USER_COL].unique().tolist())

        # Side data for beyond-accuracy metrics (method-independent).
        profiles, artist_ids = build_artist_tag_profiles(tagged)
        beyond = BeyondAccuracyInputs(
            profiles=profiles,
            item_to_row={int(a): i for i, a in enumerate(artist_ids)},
            popularity=build_item_popularity(self._train),
            n_catalogue=len(artists),
        )

        # Personalised methods first (they are the better default).
        factories = {
            "Item-item CF": ItemKNNRecommender,
            "User-user CF": UserKNNRecommender,
            "Content-based": lambda: ContentBasedRecommender(tagged),
            "Popularity (listeners)": lambda: PopularityRecommender("listeners"),
            "Popularity (plays)": lambda: PopularityRecommender("plays"),
            "Popularity (damped)": lambda: PopularityRecommender("damped"),
        }
        self.methods = list(factories)

        self._models: dict[str, BaseRecommender] = {}
        self._metrics: dict[str, dict[str, float]] = {}
        for label, factory in factories.items():
            self._models[label] = factory().fit(self._train)
            self._metrics[label] = evaluate(
                factory(), self._train, self._test, k=top_n, beyond=beyond
            )

    def _resolve(self, method: str | None) -> str:
        """Return a valid method label, falling back to the first one."""
        return method if method in self._models else self.methods[0]

    def recommend(self, method: str | None, user_id: int) -> list[Recommendation]:
        """Top-N recommendations for a user under the chosen method."""
        label = self._resolve(method)
        ids = self._models[label].recommend(user_id, n=self.top_n, exclude_seen=True)
        return [
            Recommendation(artist_id=aid, name=self._artist_name.get(aid, f"#{aid}"))
            for aid in ids
        ]

    def metrics(self, method: str | None) -> dict[str, float]:
        """Offline metrics for the chosen method."""
        return self._metrics[self._resolve(method)]

    @property
    def k(self) -> int:
        return self.top_n
