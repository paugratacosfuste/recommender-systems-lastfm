"""Compare all implemented methods on one held-out split.

Evaluates the non-personalised baselines and the collaborative-filtering models with the
shared harness and writes a comparison table to ``docs/method_comparison.csv``. This is
the running scoreboard the slide deck draws on; it grows as later modules add methods.

    python scripts/evaluate_baselines.py
"""

from __future__ import annotations

import pandas as pd

from recsys.config import DEFAULT_SEED, DEFAULT_TOP_N, RAW_DIR, USER_ARTISTS_FILE
from recsys.data.loader import load_user_artists
from recsys.data.split import leave_n_out_split
from recsys.eval.harness import compare_models
from recsys.models.cf import ItemKNNRecommender, UserKNNRecommender
from recsys.models.popularity import PopularityRecommender

DOCS_DIR = RAW_DIR.parents[1] / "docs"

METRIC_COLS = ["precision_at_k", "recall_at_k", "map_at_k", "ndcg_at_k", "fit_seconds"]


def main(k: int = DEFAULT_TOP_N) -> None:
    interactions = load_user_artists(RAW_DIR / USER_ARTISTS_FILE)
    train, test = leave_n_out_split(interactions, seed=DEFAULT_SEED)

    models = {
        "popularity_plays": PopularityRecommender(strategy="plays"),
        "popularity_listeners": PopularityRecommender(strategy="listeners"),
        "popularity_damped": PopularityRecommender(strategy="damped"),
        "item_knn": ItemKNNRecommender(),
        "user_knn": UserKNNRecommender(),
    }
    table = compare_models(models, train, test, k=k)

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out = DOCS_DIR / "method_comparison.csv"
    table.to_csv(out)

    pd.set_option("display.width", 200)
    print(f"Method comparison @ k={k} (held-out split, seed={DEFAULT_SEED}):\n")
    print(table[METRIC_COLS].round(4).to_string())
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
