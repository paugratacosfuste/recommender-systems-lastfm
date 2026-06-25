"""Compare all implemented methods on one held-out split.

Evaluates the non-personalised baselines, collaborative filtering, and content-based
filtering with the shared harness - including beyond-accuracy metrics (coverage,
diversity, novelty) - and writes ``docs/method_comparison.csv``. This is the running
scoreboard the slide deck draws on; it grows as later modules add methods.

    python scripts/evaluate_baselines.py
"""

from __future__ import annotations

import pandas as pd

from recsys.config import (
    ARTISTS_FILE,
    DEFAULT_SEED,
    DEFAULT_TOP_N,
    RAW_DIR,
    USER_ARTISTS_FILE,
    USER_TAGGED_ARTISTS_FILE,
)
from recsys.data.loader import (
    load_artists,
    load_user_artists,
    load_user_tagged_artists,
)
from recsys.data.split import leave_n_out_split
from recsys.eval.beyond_accuracy import BeyondAccuracyInputs, build_item_popularity
from recsys.eval.harness import compare_models
from recsys.models.cf import ItemKNNRecommender, UserKNNRecommender
from recsys.models.content import ContentBasedRecommender, build_artist_tag_profiles
from recsys.models.mf import ImplicitALS
from recsys.models.popularity import PopularityRecommender

DOCS_DIR = RAW_DIR.parents[1] / "docs"

METRIC_COLS = [
    "precision_at_k",
    "recall_at_k",
    "map_at_k",
    "ndcg_at_k",
    "coverage",
    "diversity",
    "novelty",
    "fit_seconds",
    "recommend_ms",
]


def main(k: int = DEFAULT_TOP_N) -> None:
    interactions = load_user_artists(RAW_DIR / USER_ARTISTS_FILE)
    tagged = load_user_tagged_artists(RAW_DIR / USER_TAGGED_ARTISTS_FILE)
    n_catalogue = len(load_artists(RAW_DIR / ARTISTS_FILE))
    train, test = leave_n_out_split(interactions, seed=DEFAULT_SEED)

    # Side data for beyond-accuracy metrics (method-independent).
    profiles, artist_ids = build_artist_tag_profiles(tagged)
    beyond = BeyondAccuracyInputs(
        profiles=profiles,
        item_to_row={int(a): i for i, a in enumerate(artist_ids)},
        popularity=build_item_popularity(train),
        n_catalogue=n_catalogue,
    )

    models = {
        "popularity_plays": PopularityRecommender(strategy="plays"),
        "popularity_listeners": PopularityRecommender(strategy="listeners"),
        "popularity_damped": PopularityRecommender(strategy="damped"),
        "item_knn": ItemKNNRecommender(),
        "user_knn": UserKNNRecommender(),
        "content_based": ContentBasedRecommender(tagged),
        "als_mf": ImplicitALS(),
    }
    table = compare_models(models, train, test, k=k, beyond=beyond)

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out = DOCS_DIR / "method_comparison.csv"
    table.to_csv(out)

    pd.set_option("display.width", 220)
    print(f"Method comparison @ k={k} (held-out split, seed={DEFAULT_SEED}):\n")
    print(table[METRIC_COLS].round(4).to_string())
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
