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
            self.__input_mapping(recommendations, interaction_historys)
        )
        return self.__compute_metric(
            transformed_recommendations, transformed_interaction_historys, k
        )

    def __input_mapping(
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
        all_items: List[str | int] = list(set(recommendations + interaction_historys))
        all_items.sort()
        # map items to an unique index
        transformed_recommendations = [
            all_items.index(item) for item in recommendations
        ]
        transformed_interaction_historys = [
            all_items.index(item) for item in interaction_historys
        ]

        return transformed_recommendations, transformed_interaction_historys

    @abstractmethod
    def __compute_metric(
        self, recommendations: List[int], interaction_historys: List[int], k: int
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
