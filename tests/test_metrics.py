"""Tests for ranking metrics. Specs for the function you implement."""

from __future__ import annotations

import pytest

from recsys.eval.metrics import precision_at_k


def test_all_relevant():
    # Every one of the top 3 is relevant -> 1.0
    assert precision_at_k([10, 20, 30], {10, 20, 30, 40}, k=3) == 1.0


def test_none_relevant():
    assert precision_at_k([1, 2, 3], {99}, k=3) == 0.0


def test_partial_relevance():
    # 2 of the top 4 are relevant -> 0.5
    assert precision_at_k([1, 2, 3, 4], {1, 3}, k=4) == pytest.approx(0.5)


def test_only_counts_top_k():
    # The relevant item at rank 4 is ignored at k=2 -> 0.0
    assert precision_at_k([1, 2, 3, 4], {4}, k=2) == 0.0


def test_fewer_than_k_recommendations_divides_by_k():
    # Only 1 item returned, it is relevant, but k=3 -> 1/3 (penalise short lists)
    assert precision_at_k([7], {7}, k=3) == pytest.approx(1 / 3)


def test_k_must_be_positive():
    with pytest.raises(ValueError):
        precision_at_k([1, 2], {1}, k=0)
