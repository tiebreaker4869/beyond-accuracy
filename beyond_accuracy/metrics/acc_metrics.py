"""
This module contains functions to compute accuracy metrics for recommender systems.
"""

import math

def hit_rate(recommendations: list[list[int]], target_items: list[list[int]], k: int) -> float:
    """
    Hit Rate is the proportion of users for whom at least one of the top-k recommendations is a hit.
    A hit is a recommended item that is also in the target items.
    Args:
        recommendations: a list of lists, where each list contains the recommended items for a user.
        target_items: a list of lists, where each list contains the target items for a user.
        k: the number of recommendations to consider.
    Returns:
        The hit rate.
    """
    hit = 0
    for recommendation, target_item in zip(recommendations, target_items):
        assert len(recommendation) >= k, "The number of recommendations is less than k."
        assert len(target_item) >= 1, "The number of target items is less than 1."
        if any(item in recommendation[:k] for item in target_item):
            hit += 1
    return hit / len(recommendations)

def ndcg(recommendations: list[list[int]], target_items: list[list[int]], k: int) -> float:
    """
    NDCG (Normalized Discounted Cumulative Gain) is a measure of ranking quality.
    It is calculated as the DCG divided by the ideal DCG.
    Args:
        recommendations: a list of lists, where each list contains the recommended items for a user.
        target_items: a list of lists, where each list contains the target items for a user.
        k: the number of recommendations to consider.
    Returns:
        The NDCG.
    """
    ndcg_sum = 0
    for recommendation, target_item in zip(recommendations, target_items):
        assert len(recommendation) >= k, "The number of recommendations is less than k."
        assert len(target_item) >= 1, "The number of target items is less than 1."
        dcg = 0
        for i, item in enumerate(recommendation[:k], start=1):
            if item in target_item:
                dcg += 1 / math.log2(i + 1)
        idcg = sum(1 / math.log2(i + 1) for i in range(1, min(k + 1, len(target_item) + 1)))
        ndcg_sum += dcg / idcg
        print(dcg, idcg)
    return ndcg_sum / len(recommendations)

def precision(recommendations: list[list[int]], target_items: list[list[int]], k: int) -> float:
    """
    Precision is the proportion of recommended items that are hits.
    Args:
        recommendations: a list of lists, where each list contains the recommended items for a user.
        target_items: a list of lists, where each list contains the target items for a user.
        k: the number of recommendations to consider.
    Returns:
        The precision.
    """
    precision_sum = 0
    for recommendation, target_item in zip(recommendations, target_items):
        assert len(recommendation) >= k, "The number of recommendations is less than k."
        assert len(target_item) >= 1, "The number of target items is less than 1."
        precision_sum += len(set(recommendation[:k]) & set(target_item)) / k
    return precision_sum / len(recommendations)

def recall(recommendations: list[list[int]], target_items: list[list[int]], k: int) -> float:
    """
    Recall is the proportion of hits that are recommended items.
    Args:
        recommendations: a list of lists, where each list contains the recommended items for a user.
        target_items: a list of lists, where each list contains the target items for a user.
        k: the number of recommendations to consider.
    Returns:
        The recall.
    """
    recall_sum = 0
    for recommendation, target_item in zip(recommendations, target_items):
        assert len(recommendation) >= k, "The number of recommendations is less than k."
        assert len(target_item) >= 1, "The number of target items is less than 1."
        recall_sum += len(set(recommendation[:k]) & set(target_item)) / len(target_item)
    return recall_sum / len(recommendations)