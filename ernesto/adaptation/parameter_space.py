import numpy as np
from scipy.spatial.distance import mahalanobis

from .region import Region, load_cluster_points
    

class ParameterSpace:
    """
    Class that defines the grid of the parameter space.
    """
    def __init__(self, 
                 parameter_space_config: list,
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
        self._regions = [Region(cluster=load_cluster_points(clusters_folder, elem['original_csv'], cols=parameter_space_config['domain_variables'] + parameter_space_config['param_variables']),
                                destination_file=output_folder / elem['original_csv'],
                                domain_variables=parameter_space_config['domain_variables'],
                                param_variables=parameter_space_config['param_variables'],
                                name=elem['name']) 
                         for elem in parameter_space_config['clusters']]
        
        self._domain_variables = parameter_space_config['domain_variables']
        self._param_variables = parameter_space_config['param_variables']
        self._active_region = None
    
    @property
    def active_region(self):
        return self._active_region
    
    def select_active_region(self, point: dict):
        """
        Select the active region based on the point.
        The distance between the domain point and regions is computed as euclidean distance.

        Args:
            point (dict): The point to check.
        """
        distance = float('inf')
        
        for region in self._regions:
            new_distance = np.linalg.norm(np.array([point.get(key, None) for key in self._domain_variables]) - region.domain_mean)
            if new_distance < distance:
                distance = new_distance
                self._active_region = region
             
    def add_params(self, params: list, region: Region):
        """
        Add the parameters to the active region.

        Args:
            params (list): The parameters to add.
        """
        if region is not None:
            return region.check_affinity_and_add(params)
        else:
            raise ValueError("No active region selected.")
        
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