"""Compute headline EDA numbers and write them to docs/eda_stats.csv.

This is the single source of truth for the figures-and-prose numbers in the report's EDA
section (tags, friendships, data quality). Run after downloading the data.

    python scripts/eda_stats.py
"""

from __future__ import annotations

import pandas as pd

from recsys.config import (
    ARTISTS_FILE,
    RAW_DIR,
    USER_ARTISTS_FILE,
    USER_FRIENDS_FILE,
    USER_TAGGED_ARTISTS_FILE,
)
from recsys.data import eda
from recsys.data.loader import (
    load_artists,
    load_user_artists,
    load_user_friends,
    load_user_tagged_artists,
)

DOCS = RAW_DIR.parents[1] / "docs"


def main() -> None:
    interactions = load_user_artists(RAW_DIR / USER_ARTISTS_FILE)
    tagged = load_user_tagged_artists(RAW_DIR / USER_TAGGED_ARTISTS_FILE)
    friends = load_user_friends(RAW_DIR / USER_FRIENDS_FILE)
    artists = load_artists(RAW_DIR / ARTISTS_FILE)

    summary = eda.summary_stats(interactions)
    quality = eda.interaction_quality(interactions)
    overlap = eda.friend_listening_overlap(interactions, friends)

    n_catalogue = len(artists)
    n_tagged = tagged["artist_id"].nunique()
    tpa = eda.tags_per_artist(tagged)
    degree = eda.friend_degree(friends)
    n_users = int(summary["n_users"])

    stats = {
        "n_users": n_users,
        "n_artists": n_catalogue,
        "n_interactions": int(summary["n_interactions"]),
        "sparsity_pct": round(summary["sparsity"] * 100, 2),
        "plays_gini": round(summary["plays_gini"], 3),
        "top100_play_share_pct": round(eda.top_k_play_share(interactions, 100) * 100, 1),
        "max_plays": int(quality["max_weight"]),
        "median_plays": int(quality["median_weight"]),
        "duplicate_pairs": int(quality["n_duplicate_pairs"]),
        "nonpositive_weights": int(quality["n_nonpositive_weight"]),
        "n_tags": int(tagged["tag_id"].nunique()),
        "n_tag_assignments": int(len(tagged)),
        "n_tagged_artists": int(n_tagged),
        "untagged_artists": int(n_catalogue - n_tagged),
        "untagged_pct": round((n_catalogue - n_tagged) / n_catalogue * 100, 1),
        "median_tags_per_artist": int(tpa.median()),
        "n_friend_edges": int(len(friends)),
        "avg_friends_per_user": round(degree.mean(), 1),
        "max_friends": int(degree.max()),
        "users_with_friends_pct": round(degree.index.nunique() / n_users * 100, 1),
        "friend_jaccard": round(overlap["friend_jaccard"], 4),
        "random_jaccard": round(overlap["random_jaccard"], 4),
        "friend_overlap_ratio": round(overlap["ratio"], 1),
    }

    out = pd.Series(stats, name="value").rename_axis("stat")
    DOCS.mkdir(parents=True, exist_ok=True)
    out.to_csv(DOCS / "eda_stats.csv")
    print(out.to_string())
    print(f"\nWrote {DOCS / 'eda_stats.csv'}")


if __name__ == "__main__":
    main()
