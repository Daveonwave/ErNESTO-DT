import numpy as np
import itertools

from .region import Cluster, load_cluster_points


class GridRegion:
    """
    
    """
    def __init__(self,
                 name: str,
                 idx: int,
                 region: list,
                 cluster: Cluster
                 ):
        """
        Args:
            idx (int): _description_
            region (dict): _description_
            cluster (Cluster): _description_
        """
        self._name = name
        self._index = idx
        self._dimensions = list(region.keys())
        self._ranges = [list(region[dim].values()) for dim in self._dimensions]
        self._cluster = cluster
        self._outliers = []

    def __repr__(self):
        return f"{self._index}-{self._name}" 
    
    @property
    def index(self):
        return self._index
    
    @property
    def bounds(self):
        return {dim: self._ranges[idx] for idx, dim in enumerate(self._dimensions)}
    
    @property
    def cluster(self):
        return self._cluster
    
    @property
    def points(self):
        return self._cluster.data_points
    
    @property
    def outliers(self):
        return self._outliers
    
    @property
    def mean(self):
        return self._cluster.centroid
    
    @property
    def covariance(self):
        return self._cluster.covariance
    
    def contains(self, point: dict):
        """
        Check if the point belong to the region.
        
        Args:
            point (dict): point in the grid that must be checked.
        """
        print(point)
        
        for var, value in point.items():
            if value < self._ranges[self._dimensions.index(var)][0] or \
                value > self._ranges[self._dimensions.index(var)][1]:
                return False
        return True
    

class ParameterSpaceGrid:
    """
    Class that defines the grid of the parameter space.
    """
    def __init__(self, 
                 grid_config: list,
                 clusters_folder: str,
                 output_folder: str = None
                 ):
        self._dimensions = list(grid_config[0]['region'].keys())
        
        assert any([elem['region'].keys() for elem in grid_config]) != self._dimensions, \
            "The dimensions of regions within the grid should be the same."
                        
        self._regions = [GridRegion(name=grid_config[idx]['name'],
                                    idx=idx, 
                                    region=elem['region'], 
                                    cluster=Cluster(data_points=load_cluster_points(clusters_folder, elem['original_cluster']), 
                                                    destination_file=output_folder / elem['destination_file'])) 
                         for idx, elem in enumerate(grid_config)]
        
        self._cur_region_idx = 0
        self._clusters_folder = clusters_folder

    @property
    def current_region(self):
        return self._regions[self._cur_region_idx]
    
    def _check_region(self, point: dict):
        """
        Check in which region the mean point of the set of is contained.
        
        Args:
            point (list): point in the grid that must be checked.
        """
        for region in self._regions:
            if region.contains(point):
                return region
        raise ValueError("The point {} is not contained in any region.".format(points))
    
    def check_region_mean(self, points: [dict]):
        """
        Check if the mean point of the set of points is contained in any region.
        
        Args:
            point (list): point in the grid that must be checked.
        """
        point = {dim: np.mean(point[dim]) for point in points for dim in self._dimensions}
        print("mean point: ", point)
        
        for region in self._regions:
            if region.contains(point):
                return region
        raise ValueError("The point {} is not contained in any region.".format(points))

    def is_region_changed(self, points: [dict]):
        """
        Returns True if the region has changed and update the index of the current region, False otherwise.
        
        Args:
            point (list): point in the grid that must be checked.
        """
        assert [list(point.keys()) == self._dimensions for point in points], \
            "Point dimensions must be {}. The list of dictionaries must contain the variables of the grid.".format(self._dimensions)
        
        new_idx = self._check_region(points)
        if self._cur_region_idx != new_idx:
            self._cur_region_idx = new_idx
            return True
        return False