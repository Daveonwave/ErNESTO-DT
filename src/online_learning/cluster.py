import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy import stats


class Cluster:
    def __init__(self):
        self._centroid = None
        self._covariance = None
        self._data_points = list()

    @property
    def data_points(self):
        return self._data_points

    @data_points.setter
    def data_points(self, value):
        self._data_points = value

    # Property for centroid
    @property
    def centroid(self):
        return self._centroid

    @centroid.setter
    def centroid(self, value):
        self._centroid = value

    # Property for covariance
    @property
    def covariance(self):
        return self._covariance

    @covariance.setter
    def covariance(self, value):
        self._covariance = value

    def add(self, point):
        self._data_points.append(point)

    def compute_centroid(self):
        if len(self.data_points) > 0:
            self.centroid = np.mean(np.array(self.data_points), axis=0)

    def compute_covariance(self):

        if len(self.data_points) > 0:
            self.covariance = np.cov(np.array(self.data_points), rowvar=False)
            if self.covariance.shape[0] == self.covariance.shape[1]:
                return self.covariance
            else:
                raise ValueError("Covariance matrix is not square.")
        else:
            raise ValueError("Cluster is empty.")

    def contains(self, point_to_test):
        point_to_test = np.array(point_to_test, dtype=float)

        if self.covariance is None:
            self.compute_covariance()

        if self.centroid is None:
            self.compute_centroid()

        try:
            inv_covariance = np.linalg.inv(self.covariance)
        except np.linalg.LinAlgError:
            # Regularize the covariance matrix if it's singular
            inv_covariance = np.linalg.inv(self.covariance + np.eye(self.covariance.shape[0]) * 1e-6)

        try:
            det = np.linalg.det(self.covariance)
            if det == 0:
                return False  # Covariance matrix is singular
        except np.linalg.LinAlgError:
            return False  # Covariance matrix is singular

        print("point to test type:", type(np.array(point_to_test)))
        print("point to test centroid:", type(self.centroid))
        print(f"Type of point_to_test: {point_to_test.dtype}, Type of centroid: {self.centroid.dtype}")

        mahalanobis_dist = mahalanobis(np.array(point_to_test), self.centroid, inv_covariance)

        t_statistic, _ = stats.ttest_1samp(np.vstack(self.data_points), point_to_test, axis=0)

        return mahalanobis_dist < np.abs(t_statistic).mean()  # t-tset non param. ?

    def update(self, point):
        self.add(point)
        self.compute_centroid()
        self.compute_covariance()
