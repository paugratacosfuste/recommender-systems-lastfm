"""Data loading, preprocessing, splitting, and exploratory statistics."""

from recsys.data.eda import gini, popularity_curve, summary_stats, top_k_play_share
from recsys.data.loader import load_artists, load_user_artists
from recsys.data.preprocess import (
    IndexMapping,
    build_index_mapping,
    build_interaction_matrix,
    confidence,
    log_scale,
)
from recsys.data.split import leave_n_out_split

__all__ = [
    "load_artists",
    "load_user_artists",
    "leave_n_out_split",
    "IndexMapping",
    "build_index_mapping",
    "build_interaction_matrix",
    "confidence",
    "log_scale",
    "gini",
    "popularity_curve",
    "summary_stats",
    "top_k_play_share",
]
