import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scipy.spatial.distance import pdist, squareform

def dunn_index(X, labels):
    distances = squareform(pdist(X))
    unique_cluster_labels = np.unique(labels)
    num_clusters = len(unique_cluster_labels)

    inter_cluster_distances = []
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            cluster_i = np.where(labels == unique_cluster_labels[i])[0]
            cluster_j = np.where(labels == unique_cluster_labels[j])[0]
            distances_ij = distances[np.ix_(cluster_i, cluster_j)]
            inter_cluster_distances.append(distances_ij.min())
    min_inter_cluster_distance = min(inter_cluster_distances)

    intra_cluster_distances = []
    for k in range(num_clusters):
        cluster_k = np.where(labels == unique_cluster_labels[k])[0]
        if len(cluster_k) > 1:
            distances_k = distances[np.ix_(cluster_k, cluster_k)]
            intra_cluster_distances.append(distances_k.max())
    max_intra_cluster_distance = max(intra_cluster_distances)

    dunn_index_value = min_inter_cluster_distance / max_intra_cluster_distance
    return dunn_index_value


def evaluate_clustering(X, labels):
    silhouette_avg = silhouette_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)
    dunn = dunn_index(X, labels)

    return {
        'Silhouette Score': silhouette_avg,
        'Calinski-Harabasz Index': calinski_harabasz,
        'Dunn Index': dunn
    }