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
        transformed_recommendations, transformed_interaction_historys = (
            self._input_mapping(recommendations, interaction_historys)
        )
        return self._compute_metric(
            transformed_recommendations, transformed_interaction_historys, k
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

        return recommendations, interaction_historys

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
