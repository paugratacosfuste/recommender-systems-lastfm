"""Beyond-accuracy metrics: coverage, intra-list diversity, novelty.

Accuracy alone rewards recommending the same safe, popular artists to everyone. These
metrics capture the qualities that make recommendations *useful and interesting*:

- **Catalogue coverage** - what fraction of all artists ever get recommended (does the
  system explore the catalogue, or hammer the same head items?).
- **Intra-list diversity** - how different the artists within one user's list are from each
  other (1 - average pairwise tag similarity).
- **Novelty** - how non-obvious the recommendations are (mean self-information; popular
  items carry little information, rare items a lot).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from recsys.config import ITEM_COL, USER_COL
from recsys.data.eda import gini


def catalogue_coverage(rec_lists: Sequence[Sequence[int]], n_catalogue: int) -> float:
    """Fraction of the catalogue that appears across all users' recommendations.

    Parameters
    ----------
    rec_lists : sequence of sequence of int
        One recommendation list per user.
    n_catalogue : int
        Total number of items in the catalogue.

    Returns
    -------
    float
        Coverage in [0, 1].
    """
    if n_catalogue <= 0:
        raise ValueError("n_catalogue must be positive")
    recommended: set[int] = set()
    for lst in rec_lists:
        recommended.update(lst)
    return len(recommended) / n_catalogue


def intra_list_diversity(
    recommended: Sequence[int],
    profiles: csr_matrix,
    item_to_row: dict[int, int],
) -> float:
    """1 - average pairwise cosine similarity of the recommended items' tag profiles.

    Items without a tag profile are skipped. A list with fewer than two profiled items has
    no pairs, so diversity is defined as 0.0.

    Parameters
    ----------
    recommended : sequence of int
        Recommended item ids.
    profiles : scipy.sparse.csr_matrix
        L2-normalised item tag profiles (so dot product == cosine similarity).
    item_to_row : dict
        Maps item id -> row index in ``profiles``.

    Returns
    -------
    float
        Intra-list diversity in [0, 1] (higher = more varied list).
    """
    rows = [item_to_row[i] for i in recommended if i in item_to_row]
    if len(rows) < 2:
        return 0.0
    sub = profiles[rows]
    sims = (sub @ sub.T).toarray()
    m = len(rows)
    off_diagonal_sum = sims.sum() - np.trace(sims)
    avg_similarity = off_diagonal_sum / (m * (m - 1))
    return float(1.0 - avg_similarity)


def novelty(recommended: Sequence[int], popularity: dict[int, float]) -> float:
    """Mean self-information ``-log2(p)`` of the recommended items.

    ``p`` is each item's popularity (fraction of users who interacted with it). Popular
    items (large p) carry little information; rare items a lot. Items absent from
    ``popularity`` (or with p <= 0) are skipped.

    Returns
    -------
    float
        Mean novelty in bits; 0.0 if no recommended item has a popularity value.
    """
    values = [
        -math.log2(popularity[i]) for i in recommended if popularity.get(i, 0.0) > 0.0
    ]
    return float(sum(values) / len(values)) if values else 0.0


def mean_recommended_popularity(
    rec_lists: Sequence[Sequence[int]], popularity: dict[int, float]
) -> float:
    """Average popularity of all recommended items (a direct popularity-bias measure).

    Higher means the system leans on already-popular artists; lower means it ventures into
    the long tail.

    Returns
    -------
    float
        Mean popularity probability across every recommended item; 0.0 if none.
    """
    values = [popularity.get(i, 0.0) for lst in rec_lists for i in lst]
    return float(sum(values) / len(values)) if values else 0.0


def recommendation_exposure_gini(
    rec_lists: Sequence[Sequence[int]], n_catalogue: int
) -> float:
    """Gini of how often each catalogue item is recommended (aggregate concentration).

    0 = every artist gets equal exposure; near 1 = recommendations pile onto a few artists
    while most are never shown. This captures popularity bias at the *system* level (across
    all users) rather than per list.

    Parameters
    ----------
    rec_lists : sequence of sequence of int
        One recommendation list per user.
    n_catalogue : int
        Catalogue size (items never recommended count as zero exposure).
    """
    if n_catalogue <= 0:
        raise ValueError("n_catalogue must be positive")
    counts = Counter(i for lst in rec_lists for i in lst)
    exposure = np.zeros(n_catalogue, dtype=float)
    values = np.array(list(counts.values()), dtype=float)
    exposure[: min(len(values), n_catalogue)] = values[:n_catalogue]
    return gini(exposure)


def build_item_popularity(interactions: pd.DataFrame) -> dict[int, float]:
    """Per-item popularity as the fraction of users who interacted with it."""
    n_users = interactions[USER_COL].nunique()
    listeners = interactions.groupby(ITEM_COL)[USER_COL].nunique()
    return (listeners / n_users).to_dict()


@dataclass(frozen=True)
class BeyondAccuracyInputs:
    """Bundle of the side data needed to score beyond-accuracy metrics.

    Attributes
    ----------
    profiles : csr_matrix
        L2-normalised item tag profiles (for intra-list diversity).
    item_to_row : dict
        item id -> row index in ``profiles``.
    popularity : dict
        item id -> popularity probability (for novelty).
    n_catalogue : int
        Catalogue size (for coverage).
    """

    profiles: csr_matrix
    item_to_row: dict[int, int]
    popularity: dict[int, float]
    n_catalogue: int
