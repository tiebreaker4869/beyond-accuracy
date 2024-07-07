"""
Accuracy related metrics
This module contains functions that compute some basic accuracy metrics for recommendation systems.
"""

from __future__ import annotations

from .base_metrics import BaseMetric

from typing import List

import numpy as np


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

        def ndcg_at_k(recommended_items, ground_truth_items, k):

            def dcg_at_k(r, k):
                r = np.asfarray(r)[:k]
                if r.size:
                    return np.sum(r / np.log2(np.arange(2, r.size + 2)))
                return 0.0

            def idcg_at_k(r, k):
                r = np.sort(np.asfarray(r))[::-1][:k]
                return dcg_at_k(r, k)

            relevance_scores = np.array(
                [1 if item in ground_truth_items else 0 for item in recommended_items]
            )
            dcg = dcg_at_k(relevance_scores, k)
            idcg = idcg_at_k(relevance_scores, k)
            ndcg = dcg / idcg if idcg > 0 else 0.0

            return ndcg

        return ndcg_at_k(recommendations, interaction_historys, k)
