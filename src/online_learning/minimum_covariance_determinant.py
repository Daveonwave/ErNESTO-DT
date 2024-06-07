from sklearn.covariance import MinCovDet
from src.online_learning import cluster as cl

def create_cluster(O, p):
    if len(O) >= p:
        mcd = MinCovDet()
        mcd.fit(O)
        cluster_mean = mcd.location_
        cluster_covariance = mcd.covariance_
        cluster = cl.Cluster()
        inlier_mask = mcd.support_
        O_bar = O[inlier_mask]

        cluster.set(O_bar)
        cluster.set_centroid(cluster_mean)
        cluster.set_variance(cluster_covariance)

        return cluster
    else:
        return None

