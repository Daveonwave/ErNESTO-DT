from sklearn.covariance import MinCovDet
from src.online_learning import cluster as cl


def create_cluster(O, p, support_fraction=0.5):
    if len(O) >= p:
        # Instantiate the Minimum Covariance Determinant estimator with support_fraction
        mcd = MinCovDet(support_fraction=support_fraction)
        try:
            # Fit the MCD estimator to the dataset O
            mcd.fit(O)

            # Extract the robust mean (location) and covariance matrix
            cluster_mean = mcd.location_
            cluster_covariance = mcd.covariance_

            # Create an instance of the Cluster class
            cluster = cl.Cluster()

            # Boolean mask indicating inliers detected by MCD
            inlier_mask = mcd.support_
            # Subset of O consisting only of inliers
            O_bar = O[inlier_mask]

            # Set inliers, centroid, and covariance matrix for the cluster
            cluster.set(O_bar)
            cluster.set_centroid(cluster_mean)
            cluster.set_covariance_matrix(cluster_covariance)

            # Return the created cluster
            return cluster
        except ValueError as e:
            print(f"An error occurred: {e}")
            return None
    else:
        # Return None if O does not meet the threshold p
        return None
