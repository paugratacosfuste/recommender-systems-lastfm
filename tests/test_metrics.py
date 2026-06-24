"""Tests for ranking metrics. Specs for the function you implement."""

from __future__ import annotations

import pytest

import math

from recsys.eval.metrics import (
    average_precision_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


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


def test_recall_finds_all_relevant():
    # Both relevant items are in the top 4 -> 2/2 = 1.0
    assert recall_at_k([1, 2, 3, 4], {1, 3}, k=4) == 1.0


def test_recall_is_capped_by_k():
    # User has 4 relevant items but only k=2 slots; best case 2/4 = 0.5
    assert recall_at_k([1, 2, 3, 4], {1, 2, 5, 6}, k=2) == pytest.approx(0.5)


def test_recall_with_no_relevant_is_zero():
    assert recall_at_k([1, 2, 3], set(), k=3) == 0.0


def test_recall_partial():
    # 1 of the user's 2 relevant items surfaced in top 3 -> 1/2
    assert recall_at_k([1, 9, 9], {1, 4}, k=3) == pytest.approx(0.5)


def test_recall_k_must_be_positive():
    with pytest.raises(ValueError):
        recall_at_k([1, 2], {1}, k=0)


def test_average_precision_rewards_early_hits():
    # hits at ranks 1 and 3: (1/1 + 2/3) / min(2,4) = 1.6667/2
    assert average_precision_at_k([1, 2, 3, 4], {1, 3}, k=4) == pytest.approx(5 / 6)


def test_average_precision_perfect_ranking_is_one():
    assert average_precision_at_k([1, 3, 2, 4], {1, 3}, k=4) == pytest.approx(1.0)


def test_average_precision_no_hits_is_zero():
    assert average_precision_at_k([5, 6], {1}, k=2) == 0.0
    assert average_precision_at_k([1, 2], set(), k=2) == 0.0


def test_ndcg_discounts_by_rank():
    # hits at ranks 1 and 3 -> dcg = 1 + 1/log2(4); idcg = 1 + 1/log2(3)
    expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k([1, 2, 3], {1, 3}, k=3) == pytest.approx(expected)


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k([1, 3, 9, 9], {1, 3}, k=4) == pytest.approx(1.0)


def test_ndcg_no_hits_is_zero():
    assert ndcg_at_k([8, 9], {1}, k=2) == 0.0
    assert ndcg_at_k([1, 2], set(), k=2) == 0.0


def test_map_ndcg_k_must_be_positive():
    with pytest.raises(ValueError):
        average_precision_at_k([1], {1}, k=0)
    with pytest.raises(ValueError):
        ndcg_at_k([1], {1}, k=0)
