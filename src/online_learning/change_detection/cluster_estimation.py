from src.online_learning.change_detection.similarity_based_clustering import mountain_method, largest_cluster
from src.online_learning.change_detection.minimum_covariance_determinant import mcv_robust_clustering
from sklearn.decomposition import PCA
from scipy import stats
import numpy as np


def cluster_estimation(cluster_data_points, outliers):

    pca_cluster = PCA(n_components=1)
    pca_outliers = PCA(n_components=1)
    principal_component_cluster = pca_cluster.fit_transform(np.array(cluster_data_points))
    principal_component_outliers = pca_outliers.fit_transform(np.array(outliers))

    alpha_c = 0.05
    ks_value, p_value = stats.ks_2samp(principal_component_cluster, principal_component_outliers)

    if p_value < alpha_c:
        print("Reject the null hypothesis: Create a new cluster.")
        centers = mountain_method(outliers=outliers, epsilon=0.001, radius=1, p=1)
        o_tilde = largest_cluster(centers=centers, outliers=outliers, epsilon=0.001)
        phi = mcv_robust_clustering(o_tilde, support_fraction=0.60)
        return phi
    else:
        print("Fail to reject the null hypothesis: No new cluster needed.")
        return None
