"""End-to-end entry point for the music recommender.

Loads the data, builds a train/test split, evaluates a representative set of methods, and
prints example recommendations for three users. This mirrors the assignment template's
``python main.py`` contract; the full module-by-module work lives in ``src/recsys`` with
deeper analysis in the notebooks, scripts, and the PDF report.

    python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

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
    WEIGHT_COL,
)
from recsys.data.loader import (  # noqa: E402
    load_artists,
    load_user_artists,
    load_user_friends,
    load_user_tagged_artists,
)
from recsys.data.split import leave_n_out_split  # noqa: E402
from recsys.eval.beyond_accuracy import (
    BeyondAccuracyInputs,
    build_item_popularity,
)  # noqa: E402
from recsys.eval.harness import compare_models  # noqa: E402
from recsys.models.cf import ItemKNNRecommender  # noqa: E402
from recsys.models.content import (
    ContentBasedRecommender,
    build_artist_tag_profiles,
)  # noqa: E402
from recsys.models.mf import ImplicitALS  # noqa: E402
from recsys.models.popularity import PopularityRecommender  # noqa: E402
from recsys.models.social import SocialRecommender  # noqa: E402


def main() -> None:
    if not (RAW_DIR / USER_ARTISTS_FILE).exists():
        print("Dataset not found. Run `python scripts/download_data.py` first.")
        return

    interactions = load_user_artists(RAW_DIR / USER_ARTISTS_FILE)
    tagged = load_user_tagged_artists(RAW_DIR / USER_TAGGED_ARTISTS_FILE)
    friends = load_user_friends(RAW_DIR / USER_FRIENDS_FILE)
    artists = load_artists(RAW_DIR / ARTISTS_FILE)
    name = dict(zip(artists[ITEM_COL], artists["name"]))

    train, test = leave_n_out_split(interactions, seed=DEFAULT_SEED)
    profiles, artist_ids = build_artist_tag_profiles(tagged)
    beyond = BeyondAccuracyInputs(
        profiles=profiles,
        item_to_row={int(a): i for i, a in enumerate(artist_ids)},
        popularity=build_item_popularity(train),
        n_catalogue=len(artists),
    )

    models = {
        "popularity": PopularityRecommender("listeners"),
        "item_item_cf": ItemKNNRecommender(),
        "content_based": ContentBasedRecommender(tagged),
        "matrix_factorization": ImplicitALS(),
        "social": SocialRecommender(friends),
    }

    print("Evaluating methods on the held-out split (this fits each model)...\n")
    table = compare_models(models, train, test, k=10, beyond=beyond)
    cols = ["precision_at_k", "ndcg_at_k", "coverage", "diversity", "novelty"]
    pd.set_option("display.width", 200)
    print(table[cols].round(4).to_string())

    print("\nExample recommendations (top 5) for three users:")
    for user in sorted(train[USER_COL].unique())[:3]:
        taste = train[train[USER_COL] == user].nlargest(3, WEIGHT_COL)[ITEM_COL].tolist()
        print(
            f"\nUser {user} - listens to: "
            + ", ".join(name.get(i, str(i)) for i in taste)
        )
        for label, model in models.items():
            recs = model.recommend(user, n=5, exclude_seen=True)
            print(f"  {label:<22} {', '.join(name.get(i, str(i)) for i in recs)}")


if __name__ == "__main__":
    main()
