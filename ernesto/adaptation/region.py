import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from scipy import stats
from scipy.stats import f


def load_cluster_points(folder: str, csv_file: str, cols:list):
    """
    Args:
        csv_file (str): _description_
    """
    data_points = []
    # Load the data points from the CSV file
    if csv_file is not None:
        data_points = pd.read_csv(folder + csv_file, usecols=cols)
    return data_points


class Region:
    def __init__(self, 
                 cluster:pd.DataFrame,
                 destination_file:str,
                 domain_variables:list,
                 param_variables:list,
                 name:str
                 ):
        """_summary_

        Args:
            cluster (list): _description_
            destination_file (str): _description_
            domain_variables (list): _description_
            param_variables (list): _description_
            name (str): _description_
        """
        self._name = name
        
        self._centroid = None
        self._covariance = None
        self._domain_mean = None
        
        self._domain_variables = domain_variables if domain_variables is not None else []
        self._param_variables = param_variables if param_variables is not None else []
        
        self._df_cluster = cluster if cluster is not None else None
        self._df_outliers = pd.DataFrame(columns=self._domain_variables + self._param_variables)
                
        self._destination_file = destination_file
        
        self.compute_centroid()
        self.compute_covariance()
        self.compute_domain_mean_point()
    
    def __repr__(self):
        return f"Region(name={self._name}, domain_mean={self._domain_mean}, centroid={self._centroid})"

    def __str__(self):
        return f"Region: {self._name}"
    
    @property
    def cluster(self):
        return self._cluster
    
    @property
    def outliers(self):
        return self._df_outliers
    
    @property
    def domain_points(self):
        return self._df_cluster[self._domain_variables].to_numpy() if self._df_cluster is not None else None

    @property
    def param_points(self):
        return self._df_cluster[self._param_variables].to_numpy() if self._df_cluster is not None else None
    
    # Property for centroid
    @property
    def centroid(self):
        return self._centroid
    
    @property
    def centroid_dict(self):
        return {param: val for param, val in zip(self._param_variables, self._centroid)}

    # Property for covariance
    @property
    def covariance(self):
        return self._covariance
    
    @property
    def domain_mean(self):
        return self._domain_mean
        
    def is_empty(self):
        return len(self._df_cluster) == 0
        
    def add(self, points: list):
        """
        Add points to the cluster.

        Args:
            points (list): list of dictionaries representing the points to be added.
        """
        if points is None:
            return
        self._df_cluster = pd.concat([self._df_cluster, pd.DataFrame.from_records(points)], ignore_index=True)
        
    def discard(self, points: list):
        """
        Add points to the outliers set.

        Args:
            points (list): list of dictionaries representing the points to be added.
        """
        if points is None:
            return
        self._df_outliers = pd.concat([self._df_outliers, pd.DataFrame.from_records(points)], ignore_index=True)
    
    def compute_domain_mean_point(self):
        """
        Compute the mean point of the domain variables in the cluster.
        
        Returns:
            np.ndarray: Mean point of the domain variables.
        """
        if self._df_cluster is not None and len(self._df_cluster) > 0:
            self._domain_mean = self._df_cluster[self._domain_variables].mean().to_numpy()
        else:
            self._domain_mean = None
    
    def compute_centroid(self):
        """
        Compute the centroid of the cluster.
        """
        if len(self._df_cluster) > 0:
            self._centroid = np.mean(self.param_points, axis=0)

    def compute_covariance(self):
        """
        Compute the covariance matrix of the cluster.

        Raises:
            ValueError: covariance matrix is not square
            ValueError: cluster is empty
        """
        if len(self._df_cluster) > 0:
            self._covariance = np.cov(np.array(self.param_points), rowvar=False)
            
            if self._covariance.shape[0] == self._covariance.shape[1]:
                return self._covariance
            else:
                raise ValueError("Covariance matrix is not square.")
        else:
            return None  # or raise an exception if you prefer
    
    def compute_covariance_inverse(self):
        """
        Compute the inverse of the covariance matrix.

        Raises:
            ValueError: covariance matrix is not square
            ValueError: cluster is empty
        """
        if self._covariance is not None and len(self._df_cluster) > 0:
            inv_covariance = np.linalg.inv(self._covariance)
            return inv_covariance
        else:
            raise ValueError("Covariance matrix is not defined or cluster is empty.")
        
    def compute_distances(self, cov_inv: np.ndarray = None):
        """
        Compute the Mahalanobis distances of the points in the cluster from the centroid.

        Args:
            cov_inv (np.ndarray, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        return [mahalanobis(x, self._centroid, cov_inv) for x in self._df_cluster[self._param_variables].values]
    
    def is_in_cluster(self, point, cov_inv, threshold):
        """
        Check if a point is within the cluster using the Mahalanobis distance.

        Args:
            point (_type_): _description_
            cov_inv (_type_): _description_
            threshold (_type_): _description_

        Returns:
            _type_: _description_
        """
        dist = mahalanobis(point, self._centroid, cov_inv)
        return dist <= threshold, dist
        
    def affinity_test(self, point: dict, alpha: float = 0.05):
        """
        Hypothesis test to check if the point is within the cluster.
        # TODO: implement a non-parametric test to check if the point is within the cluster.
        
        Args:
            point (_type_): _description_

        Returns:
            _type_: _description_
        """
        point = np.array(list(point.values()))
        covariance_inverse = self.compute_covariance_inverse()
        
        # Empirical Mahalanobis threshold based on the cluster distances
        mahal_distances = self.compute_distances(cov_inv=covariance_inverse)
        threshold = np.percentile(mahal_distances, 100 * (1 - alpha))

        is_in, distance = self.is_in_cluster(point, covariance_inverse, threshold)
        
        print(f"Threshold for Mahalanobis distance: {threshold}, Distance: {distance}")
        return is_in
        

        ## Non-parametric test: Wilcoxon signed-rank test
        # p_values = []
        # for i in range(point.shape[0]):
        #     result = stats.wilcoxon(data_points_array[:, i] - point[i], alternative='two-sided')
        #     if isinstance(result, tuple):
        #         p_value = result[1]
        #     else:
        #         p_value = result
        #     p_values.append(float(p_value))  # Ensure p_value is treated as a float

        ## Mahalanobis distance based on Fisher's F-distribution
        # p = len(self._param_variables)
        # n = len(self._df_cluster)
        # if n <= p:
        #     raise ValueError("Number of points in the cluster must be greater than the number of parameters.")
        # Quantile for the F-distribution
        #f_val = f.ppf(1 - alpha, p, n - p)
        #threshold = f_val * (p * (n**2 - 1)) / (n * (n - p))
        
        print(f"Mahalanobis distance: {mahalanobis_dist}, Threshold: {threshold}")
        
        return mahalanobis_dist <= threshold, mahalanobis_dist
    
    def check_affinity_and_add(self, points: list):
        """
        Check if the points are within the cluster and add them if they are.

        Args:
            points (list): list of dictionaries representing the points to be added.
        """
        for point in points:
            if self.affinity_test(point):
                print(f"Point {point} is in the cluster, adding to cluster.")
                self.add([point])
            else:
                print(f"Point {point} is not in the cluster, adding to outliers.")
                self.discard([point])
    
    def save(self, labels: list = None):
        """
        Save the cluster points to a CSV file.

        Args:
            folder (str): _description_
            csv_file (str): _description_
        """
        if self._df_cluster is not None:
            self._df_cluster.to_csv(self._destination_file, index=False)

