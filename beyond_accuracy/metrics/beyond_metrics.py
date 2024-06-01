"""
Metrics that go beyond accuracy.
"""

from __future__ import annotations

from .base_metrics import BaseMetric

from typing import List, Dict


class Serendipity(BaseMetric):
    """
    Serendipity metric.
    """

    def __init__(self, all_items: List[int | str]) -> None:
        super().__init__(all_items)
        self._popularity: Dict[int, int] = {}

    def fit(self, all_interactions: List[str | int]) -> None:
        _, transformed_interactions = self._input_mapping([], all_interactions)
        for item in transformed_interactions:
            if item in self._popularity:
                self._popularity[item] += 1
            else:
                self._popularity[item] = 1

    def _compute_metric(
        self,
        recommendations: List[int],
        interaction_historys: List[int],
        scores: List[float],
        k: int,
    ) -> float:
        """
        Compute the serendipity metric.

        Args:
            recommendations: List of recommended items.
            interaction_historys: List of items interacted with by the user.
            scores: List of scores for the recommended items.
            k: Number of recommendations to consider.

        Returns:
            float: The computed serendipity.
        """
        serendipity = 0.0
        for i, item in enumerate(recommendations[:k]):
            if item in interaction_historys:
                serendipity += max(
                    (scores[i] - self.__compute_popularity_based_prob(item)), 0
                )

        return serendipity / k

    def __compute_popularity_based_prob(self, item: int, alpha: int = 1) -> float:
        normaliser = sum(self._popularity.values()) + alpha * len(self._all_items)
        return (self._popularity[item] + alpha) / normaliser
