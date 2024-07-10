from typing import List, Dict, Tuple
import heapq
import numpy as np
from .base_metrics import BaseMetric


class Serendipity(BaseMetric):
    """
    Serendipity metric.
    """

    def __init__(self, all_items: List[int]) -> None:
        self._all_items = all_items
        self._popularity: Dict[int, int] = {}
        self._itemwise_metrics: Dict[int, float] = {}
        self._item_user_matrix: np.ndarray = None

    def fit_item_user_matrix(
        self, interaction_historys: Tuple[List[int], List[List[int]]]
    ) -> None:
        users, interactions = interaction_historys
        user_num = max(users) + 1
        item_num = max(self._all_items) + 1
        self._item_user_matrix = np.zeros((item_num, user_num), dtype=int)
        for user, items in zip(users, interactions):
            for item in items:
                self._item_user_matrix[item][user] += 1

    def fit(self, all_interactions: List[int]) -> None:
        for item in all_interactions:
            self._popularity[item] = self._popularity.get(item, 0) + 1

    def _compute_metric(
        self, recommendations: List[int], interaction_historys: List[int], k: int
    ) -> float:
        self._itemwise_metrics.clear()
        serendipity = 0.0
        recommendations = (recommendations + [0] * (k - len(recommendations)))[:k]

        for i, item in enumerate(recommendations):
            if item in interaction_historys:
                score = (k - i - 1) / (k - 1)
                primitive_score = self.__compute_popularity_based_prob(item, k)
                delta = max((score - primitive_score), 0)
                serendipity += delta
                self._itemwise_metrics[item] = delta
            else:
                if self._item_user_matrix is None:
                    self._itemwise_metrics[item] = 0
                else:
                    try:
                        relevance = max(
                            np.dot(
                                self._item_user_matrix[item],
                                self._item_user_matrix[interaction_item],
                            )
                            / (
                                np.linalg.norm(self._item_user_matrix[item])
                                * np.linalg.norm(
                                    self._item_user_matrix[interaction_item]
                                )
                            )
                            for interaction_item in interaction_historys
                        )
                    except Exception as e:
                        relevance = 0
                    score = (k - i - 1) / (k - 1)
                    primitive_score = self.__compute_popularity_based_prob(item, k)
                    delta = max((score - primitive_score) * relevance, 0)
                    serendipity += delta
                    self._itemwise_metrics[item] = delta

        return serendipity / k

    def __compute_popularity_based_prob(self, item: int, list_len: int) -> float:
        most_popular_items = heapq.nlargest(
            list_len, self._popularity.items(), key=lambda x: x[1]
        )
        if item not in {item[0] for item in most_popular_items}:
            return 0
        rank = (
            next(
                i
                for i, (pop_item, _) in enumerate(most_popular_items)
                if pop_item == item
            )
            + 1
        )
        return (list_len - rank) / (list_len - 1)

    def get_itemwise_metrics(self) -> Dict[int, float]:
        return self._itemwise_metrics
