import numpy as np
from joblib import Parallel, delayed
from functools import partial
from scipy.optimize import minimize
from tqdm import tqdm
from ernesto.digital_twin.bess import BatteryEnergyStorageSystem
from ernesto.adaptation.loss import *
#from cython_loss import *


class Optimizer:
    def __init__(self, 
                 battery_config: dict,
                 algorithm: str,
                 search_bounds: dict, 
                 scale_factors: dict, 
                 n_guesses: int,
                 alpha: float = 0.,
                 beta: float = 0.,
                 n_jobs: int = -1,
                 temperature_loss=False,
                 **kwargs
                 ):
        self._battery_config = battery_config
        
        self._alg = algorithm
        self._alpha = alpha
        self._beta = beta
        self._n_guesses = n_guesses    
        self._n_jobs = n_jobs
        
        self._labels = list(search_bounds.keys())
        self._bounds = [list(search_bounds[label].values()) for label in self._labels]
        
        assert self._labels == list(scale_factors.keys()), "The labels of the bounds and the scale factors must be the same."
        self._scale_factors = np.array(list(scale_factors.values()))

        self._options = {'disp': True, 'maxiter': 1000}
        if 'disp' in kwargs:
            self._options['disp'] = kwargs['disp']
        if 'maxiter' in kwargs:
            self._options['maxiter'] = kwargs['maxiter']
        if 'ftol' in kwargs:
            self._options['ftol'] = kwargs['ftol']
        if 'gtol' in kwargs:
            self._options['gtol'] = kwargs['gtol']
            
        self.temperature_loss = temperature_loss

    def _set_theta(self, battery: BatteryEnergyStorageSystem, theta: list):
        battery._electrical_model.r0.resistance, battery._electrical_model.rc.resistance, battery._electrical_model.rc.capacity = theta
    
    def lhs(self):
        n = 1
        d = len(self.bounds)
        samples = np.zeros((n, d))
        for i in range(d):
            samples[:, i] = np.random.uniform(low=self.bounds[i][0], high=self.bounds[i][1], size=n)
        for i in range(d):
            np.random.shuffle(samples[:, i])
        return samples.flatten()

    def estimate_cluster(self, init_state: dict, input_batch: dict):
        """
        Perform a step of the optimization process during the training phase to create the cluster of a region.
        This method a set of parameters that minimize the loss function.

        Args:
            init_state (dict): _description_
            input_batch (dict): _description_

        Returns:
            _type_: _description_
        """
        scaled_bounds = [(low * s, high * s) for (low, high), s in zip(self._bounds, self._scale_factors)]        
        initial_guesses = [np.array([np.random.uniform(b[0], b[1]) for b in scaled_bounds]) 
                           for _ in range(self._n_guesses)]
        
        args = (input_batch, init_state, self._battery_config, self._scale_factors, self._alpha, self._beta)
        loss = lambda x: scaled_loss(x, *args)

        results = Parallel(n_jobs=self._n_jobs)(delayed(minimize)(
            loss, x0=guess, method=self._alg, bounds=scaled_bounds, options=self._options)
                                                for guess in tqdm(initial_guesses))
        
        final_params = [(res.x / self._scale_factors).tolist() for res in results]
        return final_params
            
    
    def estimate_new_theta(self, init_state: dict, input_batch: dict, centroid: list = None):
        """
        Perform a step of the optimization process during the adaptation phase.
        This method return the best parameters found during the optimization process.
        The parameters are the ones that minimize the loss function.
        The loss function is defined in the loss.py file.

        Args:
            init_state (dict): _description_
            input_batch (dict): _description_

        Returns:
            _type_: _description_
        """
        best_value = float('inf')
        best_new_params = None
        
        scaled_bounds = [(low * s, high * s) for (low, high), s in zip(self._bounds, self._scale_factors)]    
        
        if centroid is not None:
            # Create random point following a normal centered into the scaled centroid
            initial_guesses = np.random.normal(loc=centroid * self._scale_factors, scale=0.1, size=(self._n_guesses, len(centroid)))
        else:
             # Scale the bounds according to the scale factors and create initial guesses    
            initial_guesses = [np.array([np.random.uniform(b[0], b[1]) for b in scaled_bounds]) 
                               for _ in range(self._n_guesses)]
                
        # Define the loss function with the scaled parameters
        args = (input_batch, init_state, self._battery_config, self._scale_factors, self._alpha, self._beta)
        loss = lambda x: scaled_loss(x, *args)

        results = Parallel(n_jobs=self._n_jobs)(delayed(minimize)(
            loss, x0=guess, method=self._alg, bounds=scaled_bounds, options=self._options)
                                                for guess in tqdm(initial_guesses))
        
        res = min(results, key=lambda res: res.fun)
        print(res)
        best_new_params, best_loss = res.x, res.fun
        
        print(f"New theta:{best_new_params / self._scale_factors}, loss:{best_loss}")
        return best_new_params / self._scale_factors