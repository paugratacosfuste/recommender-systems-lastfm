"""Compare all methods over several random splits and report mean +/- std.

Evaluates the popularity baselines, collaborative filtering, content-based, matrix
factorisation, and the social recommender with the shared harness - including
beyond-accuracy and popularity-bias metrics - across multiple seeds, and writes the mean
table to ``docs/method_comparison.csv`` (plus a std table). Averaging over seeds removes the
single-split fragility.

    python scripts/evaluate_baselines.py
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
from recsys.data.loader import (
    load_artists,
    load_user_artists,
    load_user_friends,
    load_user_tagged_artists,
)
from recsys.data.split import leave_n_out_split
from recsys.eval.beyond_accuracy import BeyondAccuracyInputs, build_item_popularity
from recsys.eval.harness import compare_models
from recsys.models.cf import ItemKNNRecommender, UserKNNRecommender
from recsys.models.content import ContentBasedRecommender, build_artist_tag_profiles
from recsys.models.mf import ImplicitALS
from recsys.models.popularity import PopularityRecommender
from recsys.models.social import SocialRecommender

DOCS_DIR = RAW_DIR.parents[1] / "docs"
SEEDS = [42, 7, 123, 2024, 99]

METRIC_COLS = [
    "precision_at_k",
    "recall_at_k",
    "map_at_k",
    "ndcg_at_k",
    "coverage",
    "diversity",
    "novelty",
    "popularity_bias",
    "exposure_gini",
    "fit_seconds",
    "recommend_ms",
]


def _models(tagged, friends) -> dict:
    return {
        "popularity_plays": PopularityRecommender(strategy="plays"),
        "popularity_listeners": PopularityRecommender(strategy="listeners"),
        "popularity_damped": PopularityRecommender(strategy="damped"),
        "item_knn": ItemKNNRecommender(),
        "user_knn": UserKNNRecommender(),
        "content_based": ContentBasedRecommender(tagged),
        "als_mf": ImplicitALS(),
        "social": SocialRecommender(friends),
    }


def main(k: int = 10) -> None:
    interactions = load_user_artists(RAW_DIR / USER_ARTISTS_FILE)
    tagged = load_user_tagged_artists(RAW_DIR / USER_TAGGED_ARTISTS_FILE)
    friends = load_user_friends(RAW_DIR / USER_FRIENDS_FILE)
    n_catalogue = len(load_artists(RAW_DIR / ARTISTS_FILE))
    profiles, artist_ids = build_artist_tag_profiles(tagged)
    item_to_row = {int(a): i for i, a in enumerate(artist_ids)}

    frames = []
    for seed in SEEDS:
        train, test = leave_n_out_split(interactions, seed=seed)
        beyond = BeyondAccuracyInputs(
            profiles=profiles,
            item_to_row=item_to_row,
            popularity=build_item_popularity(train),
            n_catalogue=n_catalogue,
        )
        table = compare_models(_models(tagged, friends), train, test, k=k, beyond=beyond)
        frames.append(table)
        print(f"  seed {seed} done")

    big = pd.concat(frames, keys=SEEDS, names=["seed", "method"])
    mean_df = big.groupby("method").mean().sort_values("precision_at_k", ascending=False)
    std_df = big.groupby("method").std().loc[mean_df.index]

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    mean_df.to_csv(DOCS_DIR / "method_comparison.csv")
    std_df.to_csv(DOCS_DIR / "method_comparison_std.csv")

    pd.set_option("display.width", 240)
    print(f"\nMean over {len(SEEDS)} seeds @ k={k} (the +/- columns are std):\n")
    show = pd.DataFrame(index=mean_df.index)
    for col in ["precision_at_k", "ndcg_at_k", "coverage", "diversity", "novelty"]:
        show[col] = (
            mean_df[col].round(4).astype(str) + " +/-" + std_df[col].round(4).astype(str)
        )
    print(show.to_string())
    print(f"\nWrote {DOCS_DIR / 'method_comparison.csv'} (+ _std.csv)")


if __name__ == "__main__":
    main()
