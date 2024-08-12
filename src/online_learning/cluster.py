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

        if self.covariance is None:
            self.compute_covariance()

        if self.centroid is None:
            self.compute_centroid()

        try:
            det = np.linalg.det(self.covariance)
            if det == 0:
                return False  # Covariance matrix is singular
        except np.linalg.LinAlgError:
            return False  # Covariance matrix is singular

        mahalanobis_dist = mahalanobis(np.array(point_to_test), self.centroid, np.linalg.inv(self.covariance))

        t_statistic, _ = stats.ttest_1samp(np.vstack(self.data_points), point_to_test, axis=0)

        return mahalanobis_dist < np.abs(t_statistic).mean()

    def update(self, point):
        self.add(point)
        self.compute_centroid()
        self.compute_covariance()
