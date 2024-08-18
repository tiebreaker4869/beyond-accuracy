"""
This module contains functions to compute metrics beyond accuracy such as serendipity, diversity, etc, for recommender systems.
"""

import torch
from sklearn.cluster import KMeans
import numpy as np

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
    histories_np = np.array(histories)
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

    for u in range(len(histories_np)):
        relevant_centroid_clusters = np.unique(centroids_labels[labels[histories_np[u]]])
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