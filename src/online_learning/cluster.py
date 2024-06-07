import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy import stats

class Cluster:
    def __init__(self):
        self.centroid = None
        self.covariance_matrix = None
        self.parameters = list()

    def add(self, point):
        self.parameters.append(point)

    def set(self, parameters):
        self.parameters = parameters

    def set_centroid(self, centroid):
        self.centroid = centroid

    def set_covariance_matrix(self, covariance_matrix):
        self.covariance_matrix = covariance_matrix

    def get_centroid(self):
        return self.centroid

    def get_parameters(self):
        return self.parameters

    def get_covariance_matrix(self):
        return self.covariance_matrix

    def compute_centroid(self):
        if self.parameters:
            self.centroid = np.mean(np.array(self.parameters), axis=0)

    def euclidean_distance(self, point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def compute_covariance_matrix(self):
        if self.parameters:
            flattened_cluster = [np.ravel(array) for array in self.parameters]
            stacked_cluster = np.vstack(flattened_cluster)
            self.covariance_matrix = np.cov(stacked_cluster, rowvar=False)

            if self.covariance_matrix.shape[0] == self.covariance_matrix.shape[1]:
                return self.covariance_matrix
            else:
                raise ValueError("Covariance matrix is not square.")
        else:
            raise ValueError("Cluster is empty.")

    def contains_within(self, point_to_test):
        if not self.centroid or not self.parameters:
            return False

        if self.covariance_matrix is None:
            self.compute_covariance_matrix()

        try:
            det = np.linalg.det(self.covariance_matrix)
            if det == 0:
                return False  # Covariance matrix is singular
        except np.linalg.LinAlgError:
            return False  # Covariance matrix is singular

        mahalanobis_dist = mahalanobis(np.array(point_to_test), self.centroid, np.linalg.inv(self.covariance_matrix))

        t_statistic, _ = stats.ttest_1samp(np.vstack(self.parameters), point_to_test, axis=0)

        return mahalanobis_dist < np.abs(t_statistic).mean()
