import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy import stats


def load_cluster_points(csv_file: str):
    """

    Args:
        csv_file (str): _description_
    """
    data = pd.read_csv(csv_file)
    elems = data[['r0', 'r1', 'c']].to_numpy()
    data_points = [np.array(row) for row in elems]
    return data_points


class Cluster:
    def __init__(self, 
                 data_points:list=[]
                 ):
        """_summary_

        Args:
            data_points (list, optional): _description_. Defaults to [].
        """
        self._data_points = data_points
        self._centroid = self.compute_centroid()
        self._covariance = self.compute_covariance()

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
        """
        Compute the centroid of the cluster.
        """
        if len(self._data_points) > 0:
            return np.mean(np.array(self._data_points), axis=0)

    def compute_covariance(self):
        """
        Compute the covariance matrix of the cluster.

        Raises:
            ValueError: covariance matrix is not square
            ValueError: cluster is empty
        """
        if len(self._data_points) > 0:
            self._covariance = np.cov(np.array(self._data_points), rowvar=False)
            
            if self._covariance.shape[0] == self._covariance.shape[1]:
                return self._covariance
            else:
                raise ValueError("Covariance matrix is not square.")
        else:
            raise ValueError("Cluster is empty.")

    def contains(self, point):
        """
        

        Args:
            point (_type_): _description_
        """
        point = np.array(point, dtype=float)

        if self._covariance is None:
            self.compute_covariance()

        if self._centroid is None:
            self.compute_centroid()

        try:
            inv_covariance = np.linalg.inv(self._covariance)
        except np.linalg.LinAlgError:
            # Regularize the covariance matrix if it's singular
            inv_covariance = np.linalg.inv(self._covariance + np.eye(self._covariance.shape[0]) * 1e-6)

        try:
            det = np.linalg.det(self._covariance)
            if det == 0:
                return False  # Covariance matrix is singular
        except np.linalg.LinAlgError:
            return False  # Covariance matrix is singular

        mahalanobis_dist = mahalanobis(point, self._centroid, inv_covariance)

        data_points_array = np.array(self._data_points)

        # Non-parametric test: Wilcoxon signed-rank test
        p_values = []
        for i in range(point.shape[0]):
            result = stats.wilcoxon(data_points_array[:, i] - point[i], alternative='two-sided')
            if isinstance(result, tuple):
                p_value = result[1]
            else:
                p_value = result
            p_values.append(float(p_value))  # Ensure p_value is treated as a float

        alpha = 0.05
        is_within_cluster = all(p > alpha for p in p_values)

        return mahalanobis_dist < 1.0 and is_within_cluster

    def update(self, point):
        self.add(point)
        self.compute_centroid()
        self.compute_covariance()
