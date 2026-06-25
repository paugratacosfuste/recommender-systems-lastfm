"""Evaluation metrics and the comparison harness."""

from recsys.eval.beyond_accuracy import (
    BeyondAccuracyInputs,
    build_item_popularity,
    catalogue_coverage,
    intra_list_diversity,
    mean_recommended_popularity,
    novelty,
    recommendation_exposure_gini,
)
from recsys.eval.harness import compare_models, evaluate
from recsys.eval.metrics import (
    average_precision_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

__all__ = [
    "evaluate",
    "compare_models",
    "precision_at_k",
    "recall_at_k",
    "average_precision_at_k",
    "ndcg_at_k",
    "catalogue_coverage",
    "intra_list_diversity",
    "novelty",
    "mean_recommended_popularity",
    "recommendation_exposure_gini",
    "build_item_popularity",
    "BeyondAccuracyInputs",
]
