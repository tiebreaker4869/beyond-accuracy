"""
Metrics that go beyond accuracy.
"""

from __future__ import annotations

from .base_metrics import BaseMetric

from typing import List, Dict, Tuple

import heapq

import numpy as np


class Serendipity(BaseMetric):
    """
    Serendipity metric.
    """

    def __init__(self, all_items: List[int | str]) -> None:
        super().__init__(all_items)
        self._popularity: Dict[int, int] = {}
        self._itemwise_metrics: Dict[int, float] = None
        self._item_user_matrix: List[List[int]] = None

    def fit_item_user_matrix(
        self, interaction_historys: Tuple[List[int], List[List[int]]]
    ) -> None:
        users, interactions = interaction_historys
        user_num = max(users)
        item_num = max(self._all_items)
        self._item_user_matrix = [[0] * (user_num + 1) for _ in range(item_num + 1)]
        for user, items in zip(users, interactions):
            for item in items:
                self._item_user_matrix[item][user] += 1

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
        self._itemwise_metrics = {}
        serendipity = 0.0
        # padding with 0 if recommendation list is shorter than k
        if len(recommendations) < k:
            recommendations += [0] * (k - len(recommendations))
        for i, item in enumerate(recommendations[:k]):
            if item in interaction_historys:
                score = (k - i - 1) / (k - 1)
                primitive_score = self.__compute_popularity_based_prob(item, k)
                serendipity += max((score - primitive_score), 0)
                self._itemwise_metrics[item] = max((score - primitive_score), 0)
            else:
                if self._item_user_matrix is None:
                    self._itemwise_metrics[item] = 0
                else:
                    # relevance = max(cosine_similarity(item, item_in_interaction_history))
                    relevance = 0
                    for item_in_interaction_history in interaction_historys:
                        item_vec = np.array(self._item_user_matrix[item])
                        interaction_item_vec = np.array(
                            self._item_user_matrix[item_in_interaction_history]
                        )
                        cos_similarity = np.dot(item_vec, interaction_item_vec) / (
                            np.linalg.norm(item_vec)
                            * np.linalg.norm(interaction_item_vec)
                        )
                        relevance = max(relevance, cos_similarity)
                    score = (k - i - 1) / (k - 1)
                    primitive_score = self.__compute_popularity_based_prob(item, k)
                    serendipity += max((score - primitive_score) * relevance, 0)
                    self._itemwise_metrics[item] = max(
                        (score - primitive_score) * relevance, 0
                    )
        return serendipity / k

    def __compute_popularity_based_prob(self, item: int, list_len: int) -> float:
        most_popular_items_pop = heapq.nlargest(
            list_len, self._popularity.items(), key=lambda x: x[1]
        )
        most_popular_items = [item[0] for item in most_popular_items_pop]
        if item not in most_popular_items:
            return 0
        # find the index of item in the most popular items
        rank = most_popular_items.index(item) + 1
        return (list_len - rank) / (list_len - 1)

    def get_itemwise_metrics(self) -> Dict[int, float]:
        """
        Get the item-wise serendipity metrics.

        Returns:
            Dict[int, float]: The item-wise serendipity metrics.
        """
        return self._itemwise_metrics
