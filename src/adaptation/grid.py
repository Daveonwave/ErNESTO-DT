import numpy as np
import itertools

from .cluster import Cluster


class GridRegion:
    def __init__(self,
                 idx: int,
                 var_names: list,
                 ranges: list,
                 cluster: Cluster
                 ):
        self._index = idx
        self.var_names = var_names
        self.ranges = ranges
        self.cluster = cluster
    
    def is_contained(self, point: dict):
        pass
    



class ParameterSpaceGrid:
    """
    
    """
    def __init__(self, 
                 regions: dict,
                 nominal_clusters: dict=None,
                 ranges=None, 
                 soc=None, 
                 temp=None
                 ):
        self._dimensions = list(regions.keys())        
        combinations = [element for element in itertools.product(*regions.values())]
        print(combinations)
        
        self._regions = [GridRegion(idx=idx, var_names=self._dimensions, ranges=comb) for idx, comb in enumerate(combinations)]
        
        self._ranges = ranges
        self._combinations = None
        self._create_combinations()
        self._current = self._equivalent_combination(soc, temp)

    @property
    def current(self):
        return self._current

    def _create_combinations(self):
        combinations = {}
        soc_intervals = []
        temp_intervals = []

        for key, value in self._ranges.items():
            if key.startswith('soc_interval'):
                soc_intervals.extend(value)
        print(soc_intervals)

        for key, value in self._ranges.items():
            if key.startswith('temp_interval'):
                temp_intervals.extend(value)

        index = 0
        for soc_min, soc_max in zip(soc_intervals[::2], soc_intervals[1::2]):
            for temp_min, temp_max in zip(temp_intervals[::2], temp_intervals[1::2]):
                combinations[index] = {
                    "soc_interval": (soc_min, soc_max),
                    "temp_interval": (temp_min, temp_max)
                }
                index += 1

        self._combinations = combinations

    def _equivalent_combination(self, soc, temp):
        for index, cell in self._combinations.items():
            soc_interval = cell["soc_interval"]
            temp_interval = cell["temp_interval"]
            if soc_interval[0] <= soc <= soc_interval[1] and (temp_interval[0] <= temp <= temp_interval[1]):
                return index
        return None

    def is_changed(self, soc, temp):
        new = self._equivalent_combination(soc, temp)
        if self.current == new:
            return False
        else:
            self._current = new
            return True
