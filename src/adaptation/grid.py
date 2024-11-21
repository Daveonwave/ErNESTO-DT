import numpy as np
import itertools

from .cluster import Cluster, load_cluster_points


class GridRegion:
    """
    
    """
    def __init__(self,
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
        self._index = idx
        self._dimensions = list(region.keys())
        self._ranges = [list(region[dim].values()) for dim in self._dimensions]
        self._cluster = cluster
        self._outliers = []
        
    def mean(self):
        return self._cluster.centroid
    
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
                 clusters_folder: str
                 ):
        """
        Args:
            settings (dict): _description_
        """
        self._dimensions = list(grid_config[0]['region'].keys())
        
        assert any([elem['region'].keys() for elem in grid_config]) != self._dimensions, \
            "The dimensions of regions within the grid should be the same."
                        
        self._regions = [GridRegion(idx=idx, 
                                    region=elem['region'], 
                                    cluster=Cluster(load_cluster_points(clusters_folder + elem['cluster']))) 
                         for idx, elem in enumerate(grid_config)]
        
        self._cur_region_idx = 0

    @property
    def current_region(self):
        return self._regions[self._cur_region_idx]
    
    def _check_region(self, point: dict):
        """
        Check in which region the point is contained.
        
        Args:
            point (dict): point in the grid that must be checked.
        """
        for idx, region in enumerate(self._regions):
            if region.contains(point):
                return idx
        raise ValueError("The point {} is not contained in any region.".format(point))

    def is_region_changed(self, point: dict):
        """
        Returns True if the region has changed and update the index of the current region, False otherwise.
        
        Args:
            point (dict): point in the grid that must be checked.
        """
        assert list(point.keys()) == self._dimensions, \
            "{} must be {}. The dictionary must contain the variables of the grid.".format(list(point.keys()), self._dimensions)
        
        new_idx = self._check_region(point)
        if self._cur_region_idx != new_idx:
            self._cur_region_idx = new_idx
            return True
        return False
