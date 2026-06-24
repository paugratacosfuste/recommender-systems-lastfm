"""Build cached, model-ready artifacts from the raw Last.fm files.

Reads ``data/raw/`` and writes ``data/processed/``:
  - interactions.parquet   : tidy interactions + log_weight + confidence columns
  - user_item_matrix.npz   : sparse (n_users x n_items) raw-weight matrix
  - mappings.npz           : user_ids / item_ids arrays (index == matrix row/col)

These artifacts are reproducible, so they are gitignored. Run after download_data.py:

    python scripts/build_processed.py
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import save_npz

from recsys.config import (
    ARTISTS_FILE,
    PROCESSED_DIR,
    RAW_DIR,
    USER_ARTISTS_FILE,
    WEIGHT_COL,
)
from recsys.data.eda import summary_stats
from recsys.data.loader import load_artists, load_user_artists
from recsys.data.preprocess import (
    build_index_mapping,
    build_interaction_matrix,
    confidence,
    log_scale,
)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    interactions = load_user_artists(RAW_DIR / USER_ARTISTS_FILE)
    load_artists(RAW_DIR / ARTISTS_FILE)  # validates the catalogue is present

    # Add the implicit-feedback derived columns once, here.
    interactions = interactions.assign(
        log_weight=log_scale(interactions[WEIGHT_COL]),
        confidence=confidence(interactions[WEIGHT_COL]),
    )

    mapping = build_index_mapping(interactions)
    matrix = build_interaction_matrix(interactions, mapping)

    interactions.to_parquet(PROCESSED_DIR / "interactions.parquet", index=False)
    save_npz(PROCESSED_DIR / "user_item_matrix.npz", matrix)
    np.savez(
        PROCESSED_DIR / "mappings.npz",
        user_ids=mapping.user_ids,
        item_ids=mapping.item_ids,
    )

    stats = summary_stats(interactions)
    print(f"Wrote processed artifacts to {PROCESSED_DIR}")
    print(
        f"  users={int(stats['n_users'])}  artists={int(stats['n_items'])}  "
        f"interactions={int(stats['n_interactions'])}"
    )
    print(f"  sparsity={stats['sparsity']:.4%}  plays_gini={stats['plays_gini']:.3f}")


if __name__ == "__main__":
    main()
