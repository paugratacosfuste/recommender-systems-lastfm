"""Recommender models. One algorithm family per module."""

from recsys.models.base import BaseRecommender
from recsys.models.popularity import PopularityRecommender

__all__ = ["BaseRecommender", "PopularityRecommender"]
