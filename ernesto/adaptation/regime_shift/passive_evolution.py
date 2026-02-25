import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import mahalanobis
from typing import Dict, List, Optional, Tuple, Any
import warnings

from ernesto.adaptation.base_adapter import BaseAdapter
from ernesto.adaptation.optimizer import Optimizer
from ernesto.adaptation.regime_shift.parameter_space import ParameterSpace
from ernesto.postprocessing.interactive_plot import ParameterSpaceVisualizer
from ernesto.adaptation.regime_shift.stats import *


class PassiveEvolutionRoutine(BaseAdapter):
    """
    Adaptation routine for regime shifts in the parameter space.
    """
    def __init__(self, 
                 model_config: dict,
                 sim_config: dict,
                 clusters_folder: str,
                 batch_size: int = None,
                 enable_adaptation: bool = True,
                 output_folder: str = None,
                 render: bool = False,
                 **kwargs
                 ):
        """
        Initialize the adaptation routine for regime shifts.

        Args:
            model_config (dict): Configuration for the model.
            sim_config (dict): Configuration for the simulation.
            clusters_folder (str): Path to the folder containing cluster data.
            batch_size (int, optional): _description_. Defaults to None.
            enable_adaptation (bool, optional): _description_. Defaults to True.
            output_folder (str, optional): _description_. Defaults to None.
            render (bool, optional): _description_. Defaults to False.
        """
        super().__init__(**kwargs)
        self._enable_adaptation = enable_adaptation
        self._adaptation_options = sim_config.get('adaptation', {})
        
        # Simulation variables
        self._input_batch = []
        self._batch_size = batch_size if batch_size is not None else sim_config['optimizer']['batch_size']
        self._init_state = {}
        self._current_params = None
        
        # Optimizer parameters
        self._optimizer = Optimizer(battery_config={'models_config': model_config,
                                                    'battery_options': sim_config['battery']},
                                    **sim_config['optimizer'],
                                    enabled_adaptation=self._enable_adaptation,
                                    )
        
        self._domain_variables = sim_config['parameter_space']['domain_variables']
        self._param_variables = sim_config['parameter_space']['param_variables']
        
        # Add time tracking for visualization
        self._adaptation_time_step = 0
        self._data_history = dict(zip(self.get_domain_vars() + self._param_variables + ['time'], 
                                      [list() for _ in range(len(self.get_domain_vars() + self._param_variables + ['time']))]))

    def get_domain_vars(self):
        """
        Get the domain variables of the parameter space.
        """
        return self._domain_variables

    def get_estimated_params(self):
        """
        Get the estimated parameters.
        """   
        if self._enable_adaptation:
            return self._current_params
        else:
            return {key: self._data_history[key][0] for key in self._param_variables}
    
    def _add_to_batch(self, sample: dict):
        """
        Add the sample to the batch.
        
        Args:
            sample (dict): The sample to add to the batch
        """
        self._input_batch.append(sample)
        
    def _clear_batch(self):
        """
        Clear the batch.
        """
        self._input_batch = []

    def _collect_data_points(self, points: List[float]):
        """
        Add data points to the history parameters.
        """
        for point in points:
            for key in point:
                if key not in self._data_history:
                    warnings.warn(f"Key {key} not in data history. Skipping.")
                self._data_history[key].append(point[key])

    def reset(self, **kwargs):
        """
        Reset the adapter.
        """
        self._input_batch = []
        self._init_state = {}
        self._adaptation_time_step = 0
        
        for key, value in kwargs.items():
            if key in self._data_history:
                self._data_history[key] = [value]
        
        self._current_params = {key: self._data_history[key][0] for key in self._param_variables}

    def set_init_state(self, init_state: dict):
        """
        Set the initial state of each optimization loop.
        
        Args:
            init_state (dict): The initial state of the simulation.
        """
        self._init_state = init_state

    def keep_monitoring(self):
        """
        Return True if the adapter is keeping monitoring the parameters.
        """
        return len(self._input_batch) < self._batch_size

    def track_simulation_state(self, sample: dict, domain_vars: dict):
        """
        Track the sample.
        
        Args:
            sample (dict): The sample to track.
        """
        for key in self.get_domain_vars():
            sample[key] = domain_vars[key]
            
        self._add_to_batch(sample)

    def step(self):
        """
        Enhanced step method with time series visualization support.
        """
        # Theta is associated to the mean of the domain variables in the batch      
        theta = self._optimizer.estimate_new_theta(self._init_state, self._input_batch, centroid=np.array(list(self._current_params.values())))
        mean_domain = {dim: sum(sample[dim] for sample in self._input_batch[-self._batch_size:]) / self._batch_size 
                       for dim in self._domain_variables}
        
        # Determine if point is outlier before adding to parameter space
        point_dict = {dim: val for dim, val in zip(self._param_variables, theta)}
        self._current_params = point_dict

        # Increment simulation step counter
        self._adaptation_time_step += 1
        self._collect_data_points(points=[{**point_dict, **mean_domain, 'time': self._adaptation_time_step}])
            
        self._clear_batch()
            
    def export_data_history(self, filepath: str):
        """Export time series data to CSV."""
        if not self._data_history:
            warnings.warn("No data to export.")
            return
        
        df = pd.DataFrame(self._data_history)
        df.to_csv(f"{filepath}/parameter_evolution.csv", index=False)
        
        print(f"Data exported to {filepath}")