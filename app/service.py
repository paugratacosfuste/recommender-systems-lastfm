"""Application service layer: load data, fit the available models, serve recommendations.

Built once at app startup and reused across requests. A small method registry maps each
selectable label to a fitted model plus its offline metrics (accuracy and beyond-accuracy).
Later modules plug into the same registry - the app code does not change.
"""

from __future__ import annotations

import zlib
from collections import Counter
from dataclasses import dataclass, field

import pandas as pd

from recsys.config import (
    ARTISTS_FILE,
    DEFAULT_SEED,
    DEFAULT_TOP_N,
    ITEM_COL,
    RAW_DIR,
    TAGS_FILE,
    USER_ARTISTS_FILE,
    USER_COL,
    USER_FRIENDS_FILE,
    USER_TAGGED_ARTISTS_FILE,
    WEIGHT_COL,
)
from recsys.data.loader import (
    load_artists,
    load_tags,
    load_user_artists,
    load_user_friends,
    load_user_tagged_artists,
)
from recsys.data.split import leave_n_out_split
from recsys.eval.beyond_accuracy import BeyondAccuracyInputs, build_item_popularity
from recsys.eval.harness import evaluate
from recsys.models.base import BaseRecommender
from recsys.models.cf import ItemKNNRecommender, UserKNNRecommender
from recsys.models.content import ContentBasedRecommender, build_artist_tag_profiles
from recsys.models.mf import ImplicitALS
from recsys.models.popularity import PopularityRecommender
from recsys.models.social import SocialRecommender

# One-line explanation of each method, shown in the UI.
METHOD_DESCRIPTIONS = {
    "Item-item CF": "Artists frequently co-listened with the ones you already play.",
    "Matrix factorisation (ALS)": "Learns hidden taste factors for users and artists, "
    "then matches yours to artists'.",
    "User-user CF": "Artists played by other users whose taste overlaps yours.",
    "Content-based": "Artists whose tags (genres/moods) match your listening profile.",
    "Social (friends)": "Artists your friends listen to, using the friendship graph.",
    "Popularity (listeners)": "Most distinct listeners - same list for everyone.",
    "Popularity (plays)": "Most total plays - same list for everyone.",
    "Popularity (damped)": "Popular artists with mega-hits damped - same for everyone.",
}


@dataclass
class Recommendation:
    """A single recommended (or already-played) artist for display."""

    artist_id: int
    name: str
    tags: list[str] = field(default_factory=list)
    plays: int | None = None
    initial: str = "?"
    hue: int = 0


def _build_artist_top_tags(
    tagged: pd.DataFrame, tags: pd.DataFrame, top: int = 3
) -> dict[int, list[str]]:
    """Map each artist to its ``top`` most-applied tag labels (for display)."""
    tag_label = dict(zip(tags["tag_id"], tags["tag_value"]))
    counts = tagged.groupby([ITEM_COL, "tag_id"]).size().reset_index(name="n")
    counts = counts.sort_values([ITEM_COL, "n"], ascending=[True, False])
    head = counts.groupby(ITEM_COL).head(top)
    return (
        head.groupby(ITEM_COL)["tag_id"]
        .apply(lambda s: [tag_label.get(t, "") for t in s])
        .to_dict()
    )


class RecommenderService:
    """Loads the dataset, fits every available method, and serves recommendations."""

    def __init__(self, top_n: int = DEFAULT_TOP_N) -> None:
        self.top_n = top_n
        interactions = load_user_artists(RAW_DIR / USER_ARTISTS_FILE)
        tagged = load_user_tagged_artists(RAW_DIR / USER_TAGGED_ARTISTS_FILE)
        friends = load_user_friends(RAW_DIR / USER_FRIENDS_FILE)
        artists = load_artists(RAW_DIR / ARTISTS_FILE)
        tags = load_tags(RAW_DIR / TAGS_FILE)
        self._artist_name = dict(zip(artists[ITEM_COL], artists["name"]))
        self._artist_tags = _build_artist_top_tags(tagged, tags)

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
            "Matrix factorisation (ALS)": ImplicitALS,
            "User-user CF": UserKNNRecommender,
            "Content-based": lambda: ContentBasedRecommender(tagged),
            "Social (friends)": lambda: SocialRecommender(friends),
            "Popularity (listeners)": lambda: PopularityRecommender("listeners"),
            "Popularity (plays)": lambda: PopularityRecommender("plays"),
            "Popularity (damped)": lambda: PopularityRecommender("damped"),
        }
        self.methods = list(factories)

        self._models: dict[str, BaseRecommender] = {}
        self._metrics: dict[str, dict[str, float]] = {}
        for label, factory in factories.items():
            model = factory().fit(self._train)
            self._models[label] = model
            # Reuse the fitted model for scoring (refit=False) to avoid fitting twice.
            self._metrics[label] = evaluate(
                model, self._train, self._test, k=top_n, beyond=beyond, refit=False
            )

    def _resolve(self, method: str | None) -> str:
        """Return a valid method label, falling back to the first one."""
        return method if method in self._models else self.methods[0]

    def _artist(self, artist_id: int, plays: int | None = None) -> Recommendation:
        name = self._artist_name.get(artist_id, f"#{artist_id}")
        initial = next((ch for ch in name if ch.isalnum()), "?").upper()
        return Recommendation(
            artist_id=artist_id,
            name=name,
            tags=self._artist_tags.get(artist_id, []),
            plays=plays,
            initial=initial,
            # Deterministic colour per artist for the avatar tile.
            hue=zlib.crc32(name.encode("utf-8")) % 360,
        )

    def recommend(self, method: str | None, user_id: int) -> list[Recommendation]:
        """Top-N recommendations for a user under the chosen method."""
        label = self._resolve(method)
        ids = self._models[label].recommend(user_id, n=self.top_n, exclude_seen=True)
        return [self._artist(aid) for aid in ids]

    def user_top_artists(self, user_id: int, n: int = 6) -> list[Recommendation]:
        """The user's most-played artists in training (their actual taste)."""
        rows = self._train[self._train[USER_COL] == user_id].nlargest(n, WEIGHT_COL)
        return [
            self._artist(int(r[ITEM_COL]), plays=int(r[WEIGHT_COL]))
            for _, r in rows.iterrows()
        ]

    def metrics(self, method: str | None) -> dict[str, float]:
        """Offline metrics for the chosen method."""
        return self._metrics[self._resolve(method)]

    def comparison(self, metric: str = "precision_at_k") -> list[dict]:
        """All methods' score on one metric, sorted descending (for the bar chart)."""
        rows = [
            {"label": label, "value": float(m.get(metric, 0.0))}
            for label, m in self._metrics.items()
        ]
        return sorted(rows, key=lambda r: r["value"], reverse=True)

    def user_genre_mix(self, user_id: int, n: int = 6) -> list[dict]:
        """The user's dominant genres - tag counts across the artists they play."""
        artist_ids = self._train.loc[self._train[USER_COL] == user_id, ITEM_COL]
        counts: Counter[str] = Counter()
        for aid in artist_ids:
            counts.update(self._artist_tags.get(int(aid), []))
        return [{"tag": tag, "count": c} for tag, c in counts.most_common(n)]

    def description(self, method: str | None) -> str:
        """One-line explanation of the chosen method."""
        return METHOD_DESCRIPTIONS.get(self._resolve(method), "")

    @property
    def k(self) -> int:
        return self.top_n
