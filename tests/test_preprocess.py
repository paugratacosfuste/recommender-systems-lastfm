"""Tests for preprocessing: mappings, sparse matrix, weighting."""

from __future__ import annotations

import numpy as np
import pytest

from recsys.data.preprocess import (
    build_index_mapping,
    build_interaction_matrix,
    confidence,
    log_scale,
)


def test_index_mapping_is_contiguous_and_sorted(sample_interactions):
    mapping = build_index_mapping(sample_interactions)
    assert mapping.n_users == 4
    assert mapping.n_items == 5
    # Artist ids 1..5 map to row/col indices 0..4 in sorted order.
    assert mapping.item_index(1) == 0
    assert mapping.item_index(5) == 4
    assert mapping.user_index(1) == 0


def test_unknown_id_raises(sample_interactions):
    mapping = build_index_mapping(sample_interactions)
    with pytest.raises(KeyError):
        mapping.item_index(999)


def test_matrix_shape_and_values(sample_interactions):
    mapping = build_index_mapping(sample_interactions)
    matrix = build_interaction_matrix(sample_interactions, mapping)
    assert matrix.shape == (4, 5)
    # User 1 (row 0) played artist 1 (col 0) 100 times.
    assert matrix[0, 0] == 100
    # User 1 never played artist 4 (col 3).
    assert matrix[0, 3] == 0
    # Total stored mass equals the sum of all weights.
    assert matrix.sum() == sample_interactions["weight"].sum()


def test_matrix_drops_ids_absent_from_mapping(sample_interactions):
    # Mapping built from a subset that excludes artist 5.
    subset = sample_interactions[sample_interactions["artist_id"] != 5]
    mapping = build_index_mapping(subset)
    # Building the matrix from the *full* data must silently drop artist-5 rows.
    matrix = build_interaction_matrix(sample_interactions, mapping)
    assert matrix.shape == (4, 4)
    assert matrix.sum() == subset["weight"].sum()


def test_log_scale_is_monotonic_and_compresses():
    raw = np.array([0, 1, 10, 1000])
    scaled = log_scale(raw)
    assert scaled[0] == 0.0
    assert np.all(np.diff(scaled) > 0)  # order preserved
    # Range compression: ratio shrinks dramatically after log.
    assert scaled[-1] / scaled[1] < (raw[-1] / raw[1])


def test_confidence_is_at_least_one_and_increases():
    c = confidence(np.array([1, 100, 10000]), alpha=40.0)
    assert np.all(c >= 1.0)
    assert np.all(np.diff(c) > 0)
