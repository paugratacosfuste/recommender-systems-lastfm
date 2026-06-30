"""Generate side-by-side recommendation examples for several users.

For a few example users, shows their actual top played artists (their taste) and the top-5
recommendations from each method, so the qualitative differences between algorithms are
visible. Writes ``docs/recommendation_examples.csv`` (one row per user x method), which the
notebook and the report both render - one source of truth.

    python scripts/recommendation_examples.py
"""

from __future__ import annotations

import pandas as pd

from recsys.config import (
    ARTISTS_FILE,
    DEFAULT_SEED,
    ITEM_COL,
    RAW_DIR,
    USER_ARTISTS_FILE,
    USER_COL,
    USER_FRIENDS_FILE,
    USER_TAGGED_ARTISTS_FILE,
    WEIGHT_COL,
)
from recsys.data.loader import (
    load_artists,
    load_user_artists,
    load_user_friends,
    load_user_tagged_artists,
)
from recsys.data.split import leave_n_out_split
from recsys.models.cf import ItemKNNRecommender
from recsys.models.content import ContentBasedRecommender
from recsys.models.mf import ImplicitALS
from recsys.models.popularity import PopularityRecommender
from recsys.models.social import SocialRecommender

DOCS = RAW_DIR.parents[1] / "docs"
N_EXAMPLE_USERS = 3
TOP_N = 5


def main() -> None:
    interactions = load_user_artists(RAW_DIR / USER_ARTISTS_FILE)
    tagged = load_user_tagged_artists(RAW_DIR / USER_TAGGED_ARTISTS_FILE)
    friends = load_user_friends(RAW_DIR / USER_FRIENDS_FILE)
    artists = load_artists(RAW_DIR / ARTISTS_FILE)
    name = dict(zip(artists[ITEM_COL], artists["name"]))
    train, _test = leave_n_out_split(interactions, seed=DEFAULT_SEED)

    # Three deterministic example users (smallest ids present in training).
    users = sorted(train[USER_COL].unique())[:N_EXAMPLE_USERS]

    models = {
        "Popularity (listeners)": PopularityRecommender("listeners"),
        "Item-item CF": ItemKNNRecommender(),
        "Content-based": ContentBasedRecommender(tagged),
        "ALS (MF)": ImplicitALS(),
        "Social (friends)": SocialRecommender(friends),
    }
    for model in models.values():
        model.fit(train)

    def names(ids: list[int]) -> str:
        return "; ".join(name.get(i, f"#{i}") for i in ids)

    rows = []
    for user in users:
        taste_ids = (
            train[train[USER_COL] == user].nlargest(3, WEIGHT_COL)[ITEM_COL].tolist()
        )
        taste = names(taste_ids)
        for label, model in models.items():
            recs = model.recommend(user, n=TOP_N, exclude_seen=True)
            rows.append(
                {
                    "user_id": int(user),
                    "user_taste": taste,
                    "method": label,
                    "recommendations": names(recs) if recs else "(none)",
                }
            )

    out = pd.DataFrame(rows)
    DOCS.mkdir(parents=True, exist_ok=True)
    out.to_csv(DOCS / "recommendation_examples.csv", index=False)

    for user in users:
        sub = out[out["user_id"] == user]
        print(f"\nUser {user} - listens to: {sub['user_taste'].iloc[0]}")
        for _, r in sub.iterrows():
            print(f"  {r['method']:<22} -> {r['recommendations']}")
    print(f"\nWrote {DOCS / 'recommendation_examples.csv'}")


if __name__ == "__main__":
    main()
