"""
Metrics that go beyond accuracy.
"""

from __future__ import annotations

from .base_metrics import BaseMetric

from typing import List, Dict

import heapq


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
                score = (k - i - 1) / (k - 1)
                primitive_score =  self.__compute_popularity_based_prob(item, k)
                serendipity += max(
                    (score - primitive_score), 0
                )

        return serendipity / k

    def __compute_popularity_based_prob(self, item: int, list_len: int) -> float:
        most_popular_items_pop = heapq.nlargest(list_len, self._popularity.items(), key=lambda x: x[1])
        most_popular_items = [item[0] for item in most_popular_items_pop]
        if item not in most_popular_items:
            return 0
        # find the index of item in the most popular items
        rank = most_popular_items.index(item) + 1
        return (list_len - rank) / (list_len - 1)


class OrderAwareSerendipity(BaseMetric):
    """
    Order-aware serendipity metric.
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
        k: int,
    ) -> float:
        """
        Compute the order-aware serendipity metric.

        Args:
            recommendations: List of recommended items.
            interaction_historys: List of items interacted with by the user.
            scores: List of scores for the recommended items.
            k: Number of recommendations to consider.

        Returns:
            float: The computed order-aware serendipity.
        """
        serendipity = 0.0
        relevant_items_running_count = 0
        for i, item in enumerate(recommendations[:k]):
            if item in interaction_historys:
                relevant_items_running_count += 1
                # add a order-aware factor
                score = (k - i - 1) / (k - 1)
                serendipity += (
                    max((score - self.__compute_popularity_based_prob(item, k)), 0)
                    * relevant_items_running_count
                    / (i + 1)
                )

        return serendipity / k

    def __compute_popularity_based_prob(self, item: int, list_len: int) -> float:
        most_popular_items_pop = heapq.nlargest(list_len, self._popularity.items(), key=lambda x: x[1])
        most_popular_items = [item[0] for item in most_popular_items_pop]
        if item not in most_popular_items:
            return 0
        # find the index of item in the most popular items
        rank = most_popular_items.index(item) + 1
        return (list_len - rank) / (list_len - 1)