"""Ablation: TF-IDF vs raw tag-count vectors for content-based filtering.

A "stronger grade" extension from the assignment. Both variants are scored on the same
held-out split; results go to ``docs/content_tfidf_vs_raw.csv`` (read by the report and the
notebook). This is a self-contained comparison and does not affect the main 8-method table.

    python scripts/content_variant.py
"""

from __future__ import annotations

import pandas as pd

from recsys.config import (
    DEFAULT_SEED,
    RAW_DIR,
    USER_ARTISTS_FILE,
    USER_TAGGED_ARTISTS_FILE,
)
from recsys.data.loader import load_user_artists, load_user_tagged_artists
from recsys.data.split import leave_n_out_split
from recsys.eval.harness import evaluate
from recsys.models.content import ContentBasedRecommender

DOCS = RAW_DIR.parents[1] / "docs"


def main(k: int = 10) -> None:
    interactions = load_user_artists(RAW_DIR / USER_ARTISTS_FILE)
    tagged = load_user_tagged_artists(RAW_DIR / USER_TAGGED_ARTISTS_FILE)
    train, test = leave_n_out_split(interactions, seed=DEFAULT_SEED)

    rows = []
    for label, use_tfidf in [("TF-IDF", True), ("raw counts", False)]:
        model = ContentBasedRecommender(tagged, use_tfidf=use_tfidf)
        r = evaluate(model, train, test, k=k)
        rows.append(
            {
                "variant": label,
                "precision_at_k": r["precision_at_k"],
                "recall_at_k": r["recall_at_k"],
                "ndcg_at_k": r["ndcg_at_k"],
            }
        )

    table = pd.DataFrame(rows).set_index("variant")
    DOCS.mkdir(parents=True, exist_ok=True)
    table.to_csv(DOCS / "content_tfidf_vs_raw.csv")
    print(f"Content-based, TF-IDF vs raw tag vectors @ k={k} (seed {DEFAULT_SEED}):\n")
    print(table.round(4).to_string())
    print(f"\nWrote {DOCS / 'content_tfidf_vs_raw.csv'}")


if __name__ == "__main__":
    main()
