"""Compare the non-personalised baseline strategies on the held-out split.

Evaluates the three popularity definitions (plays / listeners / damped) with the shared
evaluation harness and writes a comparison table to ``docs/baseline_comparison.csv``.
This is the Module 3 comparison floor every personalised method must beat.

    python scripts/evaluate_baselines.py
"""

from __future__ import annotations

from recsys.config import DEFAULT_SEED, DEFAULT_TOP_N, RAW_DIR, USER_ARTISTS_FILE
from recsys.data.loader import load_user_artists
from recsys.data.split import leave_n_out_split
from recsys.eval.harness import compare_models
from recsys.models.popularity import PopularityRecommender

DOCS_DIR = RAW_DIR.parents[1] / "docs"


def main(k: int = DEFAULT_TOP_N) -> None:
    interactions = load_user_artists(RAW_DIR / USER_ARTISTS_FILE)
    train, test = leave_n_out_split(interactions, seed=DEFAULT_SEED)

    models = {
        "popularity_plays": PopularityRecommender(strategy="plays"),
        "popularity_listeners": PopularityRecommender(strategy="listeners"),
        "popularity_damped": PopularityRecommender(strategy="damped"),
    }
    table = compare_models(models, train, test, k=k)

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    out = DOCS_DIR / "baseline_comparison.csv"
    table.to_csv(out)

    print(f"Baseline comparison @ k={k} (held-out split, seed={DEFAULT_SEED}):\n")
    print(table.round(4).to_string())
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
