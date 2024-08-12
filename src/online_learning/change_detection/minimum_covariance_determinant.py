from sklearn.covariance import MinCovDet
from src.online_learning.cluster import Cluster
import numpy as np


def mcv_robust_clustering(cluster, minimum_datapoints, support_fraction=0.6):
    cluster = np.array(cluster)
    if len(cluster) >= minimum_datapoints:
        mcd = MinCovDet(support_fraction=support_fraction)
        phi = Cluster()
        try:
            mcd.fit(cluster)
            mask = mcd.support_
            inliers = cluster[mask]

            phi.data_points = inliers
            phi.compute_centroid()
            phi.compute_covariance()

            return phi

        except ValueError as e:
            print(f"An error occurred: {e}")
            return None
    else:
        return None
