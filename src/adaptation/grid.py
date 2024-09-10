import numpy as np
import itertools

from .cluster import Cluster, load_cluster_points


class GridRegion:
    """
    
    """
    def __init__(self,
                 idx: int,
                 var_names: list,
                 ranges: list,
                 cluster: Cluster
                 ):
        """
        Args:
            idx (int): _description_
            var_names (list): _description_
            ranges (list): _description_
            cluster (Cluster): _description_
        """
        self._index = idx
        self._ranges = ranges
        self._cluster = cluster
        self.region_bounds = {var_names[i]: ranges[i] for i in range(len(var_names))}
    
    def is_contained(self, point: dict):
        """
        Check if the point belong to the region.
        
        Args:
            point (dict): point in the grid that must be checked.
        """
        for var, value in point.items():
            if value < self.region_bounds[var][0] or value > self.region_bounds[var][1]:
                return False
        return True
    

class ParameterSpaceGrid:
    """
    
    """
    def __init__(self, 
                 regions: dict,
                 nominal_clusters: list,
                 ):
        """
        Args:
            regions (dict): _description_
            nominal_clusters (list): _description_
        """
        self._dimensions = list(regions.keys())        
        combinations = [element for element in itertools.product(*regions.values())]
        
        assert len(combinations) == len(nominal_clusters), "The number of nominal clusters must be equal to the number of combinations."
        
        self._regions = [GridRegion(idx=idx, var_names=self._dimensions, ranges=comb, cluster=Cluster(load_cluster_points(nominal_clusters[idx]))) 
                         for idx, comb in enumerate(combinations)]
        self._cur_region_idx = 0

    @property
    def current_region(self):
        return self._cur_region_idx
    
    @property
    def cur_ranges(self):
        return self._regions[self._cur_region_idx].region_bounds

    def _check_region(self, point: dict):
        """
        Check in which region the point is contained.
        
        Args:
            point (dict): point in the grid that must be checked.
        """
        assert point.keys() == self._dimensions, "The dictionary must contain the variables of the grid."
        for idx, region in enumerate(self._regions):
            if region.is_contained(point):
                return idx
        raise ValueError("The point {} is not contained in any region.".format(point))

    def is_region_changed(self, point: dict):
        """
        Returns True if the region has changed and update the index of the current region, False otherwise.
        
        Args:
            point (dict): point in the grid that must be checked.
        """
        assert point.keys() == self._dimensions, "The dictionary must contain the variables of the grid."
        
        new_idx = self._check_region(point)
        if self._cur_region_idx != new_idx:
            self._cur_region_idx = new_idx
            return True
        return False
