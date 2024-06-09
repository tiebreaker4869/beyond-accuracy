"""
Accuracy related metrics
This module contains functions that compute some basic accuracy metrics for recommendation systems.
"""

from __future__ import annotations

from .base_metrics import BaseMetric

from typing import List

import math


class Precision(BaseMetric):
    """
    Precision metric.
    """

    def __init__(self, all_items: List[int | str]) -> None:
        super().__init__(all_items)

    def _compute_metric(
        self,
        recommendations: List[int],
        interaction_historys: List[int],
        k: int,
    ) -> float:
        """
        Compute the precision metric.

        Args:
            recommendations: List of recommended items.
            interaction_historys: List of items interacted with by the user.
            k: Number of recommendations to consider.

        Returns:
            float: The computed precision.
        """
        if k == 0:
            return 0.0

        return len(set(recommendations[:k]) & set(interaction_historys)) / k


class Recall(BaseMetric):
    """
    Recall metric.
    """

    def __init__(self, all_items: List[int | str]) -> None:
        super().__init__(all_items)

    def _compute_metric(
        self,
        recommendations: List[int],
        interaction_historys: List[int],
        k: int,
    ) -> float:
        """
        Compute the recall metric.

        Args:
            recommendations: List of recommended items.
            interaction_historys: List of items interacted with by the user.
            k: Number of recommendations to consider.

        Returns:
            float: The computed recall.
        """
        if k == 0:
            return 0.0

        return len(set(recommendations[:k]) & set(interaction_historys)) / len(
            interaction_historys
        )


class MRR(BaseMetric):
    """
    Mean Reciprocal Rank metric.
    """

    def __init__(self, all_items: List[int | str]) -> None:
        super().__init__(all_items)

    def _compute_metric(
        self,
        recommendations: List[int],
        interaction_historys: List[int],
        k: int,
    ) -> float:
        """
        Compute the Mean Reciprocal Rank metric.

        Args:
            recommendations: List of recommended items.
            interaction_historys: List of items interacted with by the user.
            k: Number of recommendations to consider.

        Returns:
            float: The computed Mean Reciprocal Rank.
        """
        for i, item in enumerate(recommendations[:k]):
            if item in interaction_historys:
                return 1 / (i + 1)
        return 0.0


class NDCG(BaseMetric):
    """
    Normalized Discounted Cumulative Gain metric.
    """

    def __init__(self, all_items: List[int | str]) -> None:
        super().__init__(all_items)

    def _compute_metric(
        self,
        recommendations: List[int],
        interaction_historys: List[int],
        k: int,
    ) -> float:
        """
        Compute the Normalized Discounted Cumulative Gain metric.

        Args:
            recommendations: List of recommended items.
            interaction_historys: List of items interacted with by the user.
            k: Number of recommendations to consider.

        Returns:
            float: The computed NDCG.
        """
        if not interaction_historys:
            return 0.0

        dcg = 0.0

        for i, item in enumerate(recommendations[:k], 1):
            if item in interaction_historys:
                dcg += 1 / math.log2(i + 1)

        idcg = sum(
            1 / math.log2(i + 1)
            for i in range(1, min(len(interaction_historys), k) + 1)
        )

        return dcg / idcg if idcg != 0 else 0.0
