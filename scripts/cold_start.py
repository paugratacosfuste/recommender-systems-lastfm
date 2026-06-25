"""Cold-start experiment: how methods cope with sparse users and unseen artists.

Two analyses:

1. Cold users - bucket test users by how much listening history they have in training, then
   measure Precision@10 per bucket per method. Memory-based CF and ALS are expected to
   degrade for low-history users, while popularity and content/social hold up better.

2. Cold items - which artists each method can even *recommend*. Collaborative filtering,
   ALS, social, and popularity can only rank artists seen in training; content-based can
   rank any tagged artist, including ones with zero listening history. This is reported as
   the size of each method's recommendable universe.

Writes a figure to docs/figures/cold_start.png.

    python scripts/cold_start.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from recsys.config import (  # noqa: E402
    ARTISTS_FILE,
    DEFAULT_SEED,
    ITEM_COL,
    RAW_DIR,
    USER_ARTISTS_FILE,
    USER_COL,
    USER_FRIENDS_FILE,
    USER_TAGGED_ARTISTS_FILE,
)
from recsys.data.loader import (  # noqa: E402
    load_artists,
    load_user_artists,
    load_user_friends,
    load_user_tagged_artists,
)
from recsys.data.split import leave_n_out_split  # noqa: E402
from recsys.eval.metrics import precision_at_k  # noqa: E402
from recsys.models.cf import ItemKNNRecommender  # noqa: E402
from recsys.models.content import ContentBasedRecommender  # noqa: E402
from recsys.models.mf import ImplicitALS  # noqa: E402
from recsys.models.popularity import PopularityRecommender  # noqa: E402
from recsys.models.social import SocialRecommender  # noqa: E402

DOCS = RAW_DIR.parents[1] / "docs"
FIG = DOCS / "figures"


def per_user_precision(model, test_relevant: dict, k: int = 10) -> dict:
    """Precision@k for each test user under an already-fitted model."""
    return {
        u: precision_at_k(model.recommend(u, n=k, exclude_seen=True), rel, k)
        for u, rel in test_relevant.items()
    }


def main() -> None:
    interactions = load_user_artists(RAW_DIR / USER_ARTISTS_FILE)
    tagged = load_user_tagged_artists(RAW_DIR / USER_TAGGED_ARTISTS_FILE)
    friends = load_user_friends(RAW_DIR / USER_FRIENDS_FILE)
    artists = load_artists(RAW_DIR / ARTISTS_FILE)
    train, test = leave_n_out_split(interactions, seed=DEFAULT_SEED)
    relevant = test.groupby(USER_COL)[ITEM_COL].apply(set).to_dict()

    # --- 1. Cold users: bucket by training-history size ----------------------------
    history = train.groupby(USER_COL).size()
    # Rank-based quartiles (robust when many users share the same history size).
    pct_rank = history.rank(method="first", pct=True)
    buckets = pd.cut(
        pct_rank,
        [0, 0.25, 0.5, 0.75, 1.0],
        labels=["Q1 (fewest)", "Q2", "Q3", "Q4 (most)"],
        include_lowest=True,
    )

    models = {
        "Item-item CF": ItemKNNRecommender(),
        "ALS (MF)": ImplicitALS(),
        "Content-based": ContentBasedRecommender(tagged),
        "Social (friends)": SocialRecommender(friends),
        "Pop (listeners)": PopularityRecommender("listeners"),
    }
    rows = []
    for name, model in models.items():
        model.fit(train)
        prec = per_user_precision(model, relevant)
        df = pd.DataFrame({"prec": prec})
        df["bucket"] = df.index.map(buckets)
        by_bucket = df.groupby("bucket", observed=True)["prec"].mean()
        for bucket, value in by_bucket.items():
            rows.append({"method": name, "bucket": bucket, "precision": value})
    pivot = pd.DataFrame(rows).pivot(index="bucket", columns="method", values="precision")

    print("Cold-user Precision@10 by training-history quartile:\n")
    print(pivot.round(4).to_string())
    print(
        f"\nHistory size per quartile boundary: {history.describe()[['min','25%','50%','75%','max']].to_dict()}"
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(marker="o", ax=ax)
    ax.set_ylabel("Precision@10")
    ax.set_xlabel("User training-history quartile (Q1 = fewest interactions)")
    ax.set_title("Cold-user behaviour: accuracy vs how much history a user has")
    ax.legend(title="method", fontsize=8)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / "cold_start.png", dpi=120)

    # --- 2. Cold items: who can recommend artists with no training plays -----------
    all_ids = set(artists[ITEM_COL])
    train_ids = set(train[ITEM_COL].unique())
    tagged_ids = set(tagged[ITEM_COL].unique())
    cold_ids = all_ids - train_ids  # zero training interactions -> invisible to CF/ALS
    content_only = cold_ids & tagged_ids  # cold but tagged -> only content can rank these
    print("\nCold-item reach (artists with zero training plays):")
    print(f"  full catalogue:                 {len(all_ids)}")
    print(f"  cold (no training interactions): {len(cold_ids)}")
    print(
        "  CF / ALS / social / popularity can recommend cold artists: 0 "
        "(not in their item space)"
    )
    print(
        f"  content-based can recommend {len(content_only)} of the cold artists "
        "(they are tagged)."
    )


if __name__ == "__main__":
    main()
