import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from ernesto.adaptation.regime_shift.stats import maha2_fisher_threshold, maha2_chi2_threshold, maha2_empirical_threshold


def load_cluster_points(folder: str, csv_file: str, cols:list):
    """
    Args:
        csv_file (str): _description_
    """
    data_points = []
    # Load the data points from the CSV file
    if csv_file is not None:
        data_points = pd.read_csv(folder + csv_file, usecols=cols)
        data_points['time'] = 0 * len(data_points)  # Initialize time to zero
    return data_points


class Region:
    def __init__(self, 
                 cluster: pd.DataFrame,
                 domain_variables: list,
                 param_variables: list,
                 creation_time: float = None,
                 name: str = " "
                 ):
        """_summary_

        Args:
            cluster (list): _description_
            domain_variables (list): _description_
            param_variables (list): _description_
            name (str): _description_
        """
        self._name = name
        
        self._centroid = None
        self._covariance = None
        self._domain_mean = None
        
        self._metric = 'fisher_mahalanobis'  # 'empirical_mahalanobis' or 'theoretical_mahalanobis'
        
        self._domain_variables = domain_variables if domain_variables is not None else []
        self._param_variables = param_variables if param_variables is not None else []
        self._creation_time = creation_time if creation_time is not None else 0.0
        
        self._df_cluster = cluster if cluster is not None else None
        #self._df_outliers = pd.DataFrame(columns=self._domain_variables + self._param_variables)
                        
        self.compute_centroid()
        self.compute_covariance()
        self.compute_domain_mean_point()
    
    def __repr__(self):
        return f"Region(name={self._name}, domain_mean={self._domain_mean}, centroid={self._centroid})"

    def __str__(self):
        return f"Region: {self._name}"
    
    @property
    def name(self):
        return self._name

    @property
    def cluster(self):
        return self._df_cluster
    
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
    
    def compute_inverse_covariance(self):
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
        
    def affinity_test(self, points: list, alpha: float = 0.05):
        """
        Hypothesis test to check if the point is within the cluster.
        # TODO: implement a non-parametric test to check if the point is within the cluster.
        
        Args:
            point (_type_): _description_

        Returns:
            _type_: _description_
        """
        points = np.array(points)
        
        # Empirical Mahalanobis distance based on the cluster distances
        if self._metric == 'empirical_mahalanobis':
            results = maha2_empirical_threshold(X=points, mean=self._centroid, cov=self._covariance, alpha=alpha)

        # Theoretical Mahalanobis distance based on Fisher distribution
        elif self._metric == 'fisher_mahalanobis':
            results = maha2_fisher_threshold(X=points, mean=self._centroid, cov=self._covariance, n=len(self._df_cluster), alpha=alpha)

        # Chi-squared Mahalanobis distance based on Chi-squared distribution
        elif self._metric == 'chi2_mahalanobis':
            results = maha2_chi2_threshold(X=points, mean=self._centroid, cov=self._covariance, alpha=alpha)

        else:
            raise ValueError("Metric not recognized. Use 'empirical_mahalanobis' or 'theoretical_mahalanobis'.")

        return results
    
    def save(self, filepath: str):
        """
        Save the cluster points to a CSV file.

        Args:
            folder (str): _description_
            csv_file (str): _description_
        """
        if self._df_cluster is not None:
            self._df_cluster.to_csv(filepath, index=False)

