"""
This module contains functions to compute metrics beyond accuracy such as serendipity, diversity, etc, for recommender systems.
"""

import torch
from sklearn.cluster import KMeans
import numpy as np
from collections import defaultdict

def cluster_based_serendipity(histories: list[list[int]], recommendations: list[list[int]], item_embedding: torch.Tensor, k: int) -> float:
    """
    Serendipity metrics proposed in https://arxiv.org/pdf/2305.11044
    Args:
        - histories: list of list of int, the list of user histories
        - recommendations: list of list of int, the list of recommendations
        - item_embedding: torch.Tensor, the item embeddings
        - k: int, the number of recommendations to consider
    Returns:
        - float, the serendipity score
    """
    recommendations_np = np.array(recommendations)
    # cluster movie embeddings
    clustering = KMeans(n_clusters=int(len(item_embedding)**0.5)).fit(item_embedding)
    labels = clustering.labels_

    # calculate cluster centroids
    cluster_centroids = torch.zeros((len(np.unique(labels)), item_embedding.shape[1]))
    for c in np.unique(labels):
        cluster_centroids[c] = torch.mean(item_embedding[labels == c], axis=0)
    
    # cluster the centroids again
    centroids_clustering = KMeans(n_clusters=int(len(cluster_centroids)**0.5)).fit(cluster_centroids)
    centroids_labels = centroids_clustering.labels_
    unique_cl_counts = np.unique(centroids_labels, return_counts=True)[1]

    top_k = recommendations_np[:, :k]
    res = []

    for u in range(len(histories)):
        relevant_centroid_clusters = np.unique(centroids_labels[labels[histories[u]]])
        n_relevant_cluster = sum(unique_cl_counts[relevant_centroid_clusters])
        
        top_k_labels = labels[top_k[u]]
        unique_c = []
        num_ruc = 0
        
        for c in top_k_labels:
            cl = centroids_labels[c]
            if cl in relevant_centroid_clusters and c not in unique_c:
                num_ruc += 1
                unique_c.append(c)
        
        min_val = min(n_relevant_cluster, k)
        val = num_ruc / min_val
        res.append(val)
    
    diversity_score = sum(res) / len(res)
    
    return diversity_score

class Serendipity:

    def __init__(self, histories: list[list[int]], recommendations: list[list[int]], target_items: list[list[int]], k: int):
        self.recommendations = recommendations
        self.target_items = target_items
        self.k = k
        self.histories = histories
        self.popularity = None
        self.k_most_popular: list = None
        self._fit()
    
    def _fit(self):
        # calculate the popularity of each item in histories
        self.popularity = defaultdict(int)
        for hist in self.histories:
            for item in hist:
                self.popularity[item] += 1
        self.k_most_popular = sorted(self.popularity, key=self.popularity.get, reverse=True)[:self.k]
    
    def unexpectedness_based_serendipity(self) -> float:
        """
        Measure the serendipity of the recommendations as the product of unexpectedness and relevance.
        Args:
            - None
        Returns:
            - float, the serendipity score
        """
        serendipity = 0
        for recommendation, target in zip(self.recommendations, self.target_items):
            current_serendipity = 0
            for item in recommendation[:self.k]:
                if item in target:
                    rank = recommendation.index(item) + 1
                    p_iu = (self.k - rank) / (self.k - 1)
                    rank_pop = self.k_most_popular.index(item) + 1 if item in self.k_most_popular else self.k
                    p_pop = (self.k - rank_pop) / (self.k - 1)
                    current_serendipity += max(p_iu - p_pop, 0)
            serendipity += current_serendipity / self.k
        return serendipity / len(self.recommendations)
    
    def unexpected_based_orderaware_serendipity(self) -> float:
        """
        Measure the serendipity of the recommendations as the product of unexpectedness and relevance.
        Also consider the order of the recommendations.
        Args:
            - None
        Returns:
            - float, the serendipity score
        """
        serendipity = 0
        for recommendation, target in zip(self.recommendations, self.target_items):
            current_serendipity = 0
            relevant_count = 0
            for idx, item in enumerate(recommendation[:self.k], start = 1):
                if item in target:
                    relevant_count += 1
                    rank = recommendation.index(item) + 1
                    p_iu = (self.k - rank) / (self.k - 1)
                    rank_pop = self.k_most_popular.index(item) + 1 if item in self.k_most_popular else self.k
                    p_pop = (self.k - rank_pop) / (self.k - 1)
                    current_serendipity += max(p_iu - p_pop, 0) * relevant_count / idx
            serendipity += current_serendipity / self.k
        return serendipity / len(self.recommendations)
