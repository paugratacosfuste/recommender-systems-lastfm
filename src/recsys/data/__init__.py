"""Data loading, preprocessing, splitting, and exploratory statistics."""

from recsys.data.eda import (
    friend_degree,
    friend_listening_overlap,
    gini,
    interaction_quality,
    most_active_users,
    popularity_curve,
    summary_stats,
    tag_frequency,
    tags_per_artist,
    top_k_play_share,
)
from recsys.data.loader import (
    load_artists,
    load_tags,
    load_user_artists,
    load_user_friends,
    load_user_tagged_artists,
)
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
    "load_tags",
    "load_user_tagged_artists",
    "load_user_friends",
    "leave_n_out_split",
    "IndexMapping",
    "build_index_mapping",
    "build_interaction_matrix",
    "confidence",
    "log_scale",
    "gini",
    "most_active_users",
    "popularity_curve",
    "summary_stats",
    "top_k_play_share",
    "interaction_quality",
    "tag_frequency",
    "tags_per_artist",
    "friend_degree",
    "friend_listening_overlap",
]
