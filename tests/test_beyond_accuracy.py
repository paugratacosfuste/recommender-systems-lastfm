"""Tests for beyond-accuracy metrics."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from recsys.eval.beyond_accuracy import (
    build_item_popularity,
    catalogue_coverage,
    intra_list_diversity,
    novelty,
)


def test_catalogue_coverage_counts_unique_items():
    # Two users recommended items {1,2,3} and {3,4}; union = 4 of 10 -> 0.4
    assert catalogue_coverage([[1, 2, 3], [3, 4]], n_catalogue=10) == pytest.approx(0.4)


def test_catalogue_coverage_requires_positive_catalogue():
    with pytest.raises(ValueError):
        catalogue_coverage([[1]], n_catalogue=0)


def _profiles():
    # Rows 0,1 identical (cos 1); row 2 orthogonal to them. Unit-normalised.
    m = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    return csr_matrix(m), {10: 0, 11: 1, 12: 2}


def test_diversity_identical_items_is_zero():
    profiles, item_to_row = _profiles()
    # Two identical-profile artists -> no variety -> diversity 0.
    assert intra_list_diversity([10, 11], profiles, item_to_row) == pytest.approx(0.0)


def test_diversity_orthogonal_items_is_one():
    profiles, item_to_row = _profiles()
    # Orthogonal artists -> maximally diverse -> 1.
    assert intra_list_diversity([10, 12], profiles, item_to_row) == pytest.approx(1.0)


def test_diversity_needs_two_profiled_items():
    profiles, item_to_row = _profiles()
    assert intra_list_diversity([10], profiles, item_to_row) == 0.0
    # Item with no profile is skipped, leaving < 2 -> 0.
    assert intra_list_diversity([10, 999], profiles, item_to_row) == 0.0


def test_novelty_is_mean_self_information():
    # p=0.5 -> -log2(0.5)=1 bit; p=0.25 -> 2 bits; mean = 1.5
    pop = {1: 0.5, 2: 0.25}
    assert novelty([1, 2], pop) == pytest.approx(1.5)


def test_novelty_skips_unknown_items():
    pop = {1: 0.5}
    assert novelty([1, 999], pop) == pytest.approx(1.0)
    assert novelty([999], pop) == 0.0


def test_build_item_popularity_fraction_of_users():
    import pandas as pd

    from recsys.config import ITEM_COL, USER_COL, WEIGHT_COL

    df = pd.DataFrame(
        {
            USER_COL: [1, 2, 3, 1],
            ITEM_COL: [10, 10, 20, 20],
            WEIGHT_COL: [5, 5, 5, 5],
        }
    )
    pop = build_item_popularity(df)
    # Item 10 has 2 of 3 users; item 20 has 2 of 3 users.
    assert pop[10] == pytest.approx(2 / 3)
    assert pop[20] == pytest.approx(2 / 3)
