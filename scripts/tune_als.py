"""Grid-search ALS hyperparameters on a validation split carved from training.

To avoid tuning on the test set, the training interactions are split again into an inner
train and a validation set; ALS is scored on validation, and the winning configuration is
reported (and then set as the ImplicitALS default). Run:

    python scripts/tune_als.py
"""

from __future__ import annotations

import itertools

import pandas as pd

from recsys.config import DEFAULT_SEED, RAW_DIR, USER_ARTISTS_FILE
from recsys.data.loader import load_user_artists
from recsys.data.split import leave_n_out_split
from recsys.eval.harness import evaluate
from recsys.models.mf import ImplicitALS

# Kept modest so the search runs in a couple of minutes; the validation slice also makes
# each fit cheaper than on the full training set.
FACTORS = [64, 96]
REG = [0.01, 0.05, 0.1]
ALPHA = [40.0]
ITERATIONS = [15]


def main() -> None:
    interactions = load_user_artists(RAW_DIR / USER_ARTISTS_FILE)
    train_full, _test = leave_n_out_split(interactions, seed=DEFAULT_SEED)
    # Inner split for tuning - the held-out test set is never touched here.
    inner_train, val = leave_n_out_split(train_full, seed=0)

    results = []
    for f, reg, alpha, iters in itertools.product(FACTORS, REG, ALPHA, ITERATIONS):
        model = ImplicitALS(
            factors=f,
            regularization=reg,
            alpha=alpha,
            iterations=iters,
            seed=DEFAULT_SEED,
        )
        r = evaluate(model, inner_train, val, k=10)
        results.append(
            {
                "factors": f,
                "reg": reg,
                "alpha": alpha,
                "iters": iters,
                "val_P@10": r["precision_at_k"],
                "val_NDCG": r["ndcg_at_k"],
                "fit_s": r["fit_seconds"],
            }
        )
        print(
            f"factors={f} reg={reg} alpha={alpha} iters={iters}: "
            f"val P@10={r['precision_at_k']:.4f} NDCG={r['ndcg_at_k']:.4f} "
            f"fit={r['fit_seconds']:.1f}s"
        )

    table = pd.DataFrame(results).sort_values("val_P@10", ascending=False)
    best = table.iloc[0]
    print("\nValidation grid (best first):")
    print(table.to_string(index=False))
    print(
        f"\nBEST: factors={int(best['factors'])} reg={best['reg']} "
        f"alpha={best['alpha']} iters={int(best['iters'])} "
        f"(val P@10={best['val_P@10']:.4f})"
    )


if __name__ == "__main__":
    main()
