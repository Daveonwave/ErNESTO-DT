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
    print('Len of first PC for cluster datapoints:', len(principal_component_cluster))
    print('type of datapoints:', type(principal_component_cluster))
    print('shape of datapoints:', principal_component_cluster.shape)
    print('Len of first PC for outliers:', len(principal_component_outliers))
    print('type of outliers:', type(principal_component_outliers))
    print('shape of outliers:', principal_component_outliers.shape)
    cluster_flat = principal_component_cluster.flatten()
    outliers_flat = principal_component_outliers.flatten()



    alpha_c = 0.05
    ks_value, p_value = stats.ks_2samp(cluster_flat, outliers_flat)

    if p_value < alpha_c:
        print("Reject the null hypothesis: Create a new cluster.")
        centers = mountain_method(outliers=outliers, epsilon=0.001, radius=1, p=1)
        o_tilde = largest_cluster(centers=centers, outliers=outliers, epsilon=0.001)
        print('len of o_tilde', len(o_tilde))
        phi = mcv_robust_clustering(cluster=o_tilde, minimum_datapoints=5, support_fraction=0.50)
        return phi
    else:
        print("Fail to reject the null hypothesis: No new cluster needed.")
        return None
