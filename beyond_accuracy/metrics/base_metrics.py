"""
Abstract base class for metrics
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from typing import List, Tuple


class BaseMetric(ABC):
    """
    Abstract base class for metrics to be used in recommendation systems.
    """

    def __init__(self, all_items: List[int | str]) -> None:
        self._all_items = all_items

    def compute(
        self,
        recommendations: List[str | int],
        interaction_historys: List[str | int],
        scores: List[float],
        k: int,
    ) -> float:
        """
        Compute the metric for the given recommendations and interaction history.

        Args:
            recommendations: List of recommended items.
            interaction_historys: List of items interacted with by the user.
            scores: List of scores for the recommended items.
            k: Number of recommendations to consider.

        Returns:
            float: The computed metric.
        """
        transformed_recommendations, transformed_interaction_historys = (
            self._input_mapping(recommendations, interaction_historys)
        )
        return self._compute_metric(
            transformed_recommendations, transformed_interaction_historys, scores, k
        )

    def _input_mapping(
        self, recommendations: List[str | int], interaction_historys: List[str | int]
    ) -> Tuple[List[int], List[int]]:
        """
        Map the input lists to integers.

        Args:
            recommendations: List of recommended items.
            interaction_historys: List of items interacted with by the user.

        Returns:
            Tuple[List[int], List[int]]: The mapped lists.
        """
        self._all_items.sort()
        # map items to an unique index
        transformed_recommendations = [
            self._all_items.index(item) for item in recommendations
        ]
        transformed_interaction_historys = [
            self._all_items.index(item) for item in interaction_historys
        ]

        return transformed_recommendations, transformed_interaction_historys

    @abstractmethod
    def _compute_metric(
        self,
        recommendations: List[int],
        interaction_historys: List[int],
        scores: List[float],
        k: int,
    ) -> float:
        """
        Compute the metric for the given recommendations and interaction history.

        Args:
            recommendations: List of recommended items.
            interaction_historys: List of items interacted with by the user.
            k: Number of recommendations to consider.

        Returns:
            float: The computed metric.
        """
        pass
