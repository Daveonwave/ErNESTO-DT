import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy import stats

def load_cluster_points(folder: str, csv_file: str):
    """
    Args:
        csv_file (str): _description_
    """
    data_points = []
    # Load the data points from the CSV file
    if csv_file is not None:
        data = pd.read_csv(folder + csv_file)
        elems = data[['r0', 'r1', 'c1']].to_numpy()
        data_points = [np.array(row) for row in elems]
    return data_points


class Cluster:
    def __init__(self, 
                 data_points:list=[],
                 destination_file:str=None,
                 ):
        """_summary_

        Args:
            data_points (list, optional): _description_. Defaults to [].
        """
        self._data_points = data_points
        self._centroid = []
        self._covariance = []
        
        self.compute_centroid()
        self.compute_covariance()
        
        self._destination_file = destination_file

    @property
    def data_points(self):
        return self._data_points

    # Property for centroid
    @property
    def centroid(self):
        return self._centroid

    # Property for covariance
    @property
    def covariance(self):
        return self._covariance
        
    def is_empty(self):
        return len(self._data_points) == 0
        
    def add(self, points: list):
        """
        Add points to the cluster.

        Args:
            points (list): _description_
        """
        if points is None:
            return
        
        if isinstance(points[0], list):
            [self._data_points.append(point) for point in points]
        else:
            self._data_points.append(points)     
    
    def compute_centroid(self):
        """
        Compute the centroid of the cluster.
        """
        if len(self._data_points) > 0:
            self._centroid = np.mean(self._data_points, axis=0)

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
            return None  # or raise an exception if you prefer

    def contains(self, point):
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
    
    def save(self, labels: list = None):
        """
        Save the cluster points to a CSV file.

        Args:
            folder (str): _description_
            csv_file (str): _description_
        """
        if self._data_points:
            data = pd.DataFrame.from_records(self.data_points, columns=labels)
            data.to_csv(self._destination_file, index=False)

