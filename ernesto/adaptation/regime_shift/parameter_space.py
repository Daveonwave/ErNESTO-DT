import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis

from .region import Region, load_cluster_points
    

class ParameterSpace:
    """
    Class that defines the grid of the parameter space.
    """
    def __init__(self, 
                 parameter_space_config: dict,
                 clusters_folder: str,
                 output_folder: str = None
                 ):
        """
        Initialize the parameter space with the configuration and the clusters.

        Args:
            parameter_space_config (list): _description_
            clusters_folder (str): _description_
            output_folder (str, optional): _description_. Defaults to None.
        """
        self._regions = [Region(cluster=load_cluster_points(folder=clusters_folder, csv_file=elem['original_csv'], cols=parameter_space_config['domain_variables'] + parameter_space_config['param_variables']),
                                domain_variables=parameter_space_config['domain_variables'],
                                param_variables=parameter_space_config['param_variables'],
                                name=elem['name']) 
                         for elem in parameter_space_config['clusters']]
        
        self._domain_variables = parameter_space_config['domain_variables']
        self._param_variables = parameter_space_config['param_variables']
        self._active_region = None
        self._cluster_selection_mode = parameter_space_config.get('cluster_selection_mode', 'latest')  # 'closest' or 'latest'
        
        self._df_outliers = None
        # Load outliers if specified (useful to test the faulty cluster creation)
        if parameter_space_config['outliers']:
            self._df_outliers = load_cluster_points(folder=clusters_folder, 
                                                    csv_file=parameter_space_config['outliers'], 
                                                    cols=parameter_space_config['domain_variables'] + parameter_space_config['param_variables'] + ['time'])
    @property
    def param_variables(self):
        return self._param_variables

    @property
    def domain_variables(self):
        return self._domain_variables

    @property
    def active_region(self):
        return self._active_region
    
    @property
    def clusters(self):
        return [region.cluster for region in self._regions]
    
    @property
    def regions(self):
        return self._regions

    @property
    def outliers(self):
        return self._df_outliers

    def add_cluster(self, points: pd.DataFrame, curr_time: float, name: str):
        """
        Add a new cluster to the parameter space.

        Args:
            points (list): The points of the new cluster.
        """
        self._regions.append(Region(cluster=points,
                                     domain_variables=self._domain_variables,
                                     param_variables=self._param_variables,
                                     creation_time=curr_time,
                                     name=name))

    def select_active_region(self, point: dict = None):
        """
        Select the active region based on the point.
        The distance between the domain point and regions is computed as euclidean distance.

        Args:
            point (dict): The point to check if 'closest' mode is selected.
        """
        # Select the latest created cluster
        if self._cluster_selection_mode == 'latest':
            self._active_region = max(self.regions, key=lambda region: region._creation_time)
        
        # Select the closest cluster in the domain space
        elif self._cluster_selection_mode == 'closest':
            distance = float('inf')

            if self._active_region is not None:
                distance = np.linalg.norm(np.array([point.get(key, None) for key in self._domain_variables]) - self._active_region.domain_mean)

            for region in self._clusters:
                new_distance = np.linalg.norm(np.array([point.get(key, None) for key in self._domain_variables]) - region.domain_mean)
                if new_distance < distance:
                    distance = new_distance
                    self._active_region = region
        else:
            raise ValueError(f"Unknown cluster selection mode: {self._cluster_selection_mode}. Supported modes are 'closest' and 'latest'.")
             
    def eval_estimated_params(self, params: list, region: Region, mean_domain: dict, curr_time: float):
        """
        Add the parameters to the active region.

        Args:
            params (list): The parameters to add.
        """ 
        if region is not None:
            for point in params:
                d, threshold, p, is_outlier = region.affinity_test(points=[list(point.values())], alpha=0.05)
                d, p, is_outlier = d[0], p[0], is_outlier[0]
                print(f"Distance: {d}, Threshold: {threshold}, p-value: {p}, Is in cluster: {not is_outlier}")
                print("=================================\n")
                
                # Add the mean of the domain variables to the point
                point.update(mean_domain)
                point['time'] = curr_time

                if not is_outlier:
                    # Point is in the cluster, so we add it to the cluster
                    region.add([point])
                    region.compute_centroid()
                    region.compute_covariance()
                else:
                    # Point is an outlier, so we add it to the outliers
                    if self._df_outliers is None:
                        self._df_outliers = pd.DataFrame.from_records([point], columns=self._domain_variables + self._param_variables + ['time'])
                    else:
                        self._df_outliers = pd.concat(
                            [self._df_outliers, pd.DataFrame.from_records([point])],
                            ignore_index=True
                        )
        else:
            raise ValueError("No active region selected.")

        return is_outlier

    def check_batch_mean_domain(self, input_batch: list, window_size: int = None):
        """
        Check the mean of the domain variables in the input batch.

        Args:
            input_batch (list): _description_
            window_size (int, optional): _description_. Defaults to None.
        """
        if window_size is None:
            window_size = len(input_batch)
        
        mean_domain = {dim: sum(sample[dim] for sample in input_batch[-window_size:]) / window_size 
                       for dim in self._domain_variables}
        return mean_domain