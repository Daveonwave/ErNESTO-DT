import numpy as np
import logging
from joblib import Parallel, delayed
from functools import partial
from scipy.optimize import minimize
from ernesto.digital_twin.bess import BatteryEnergyStorageSystem
from ernesto.adaptation.loss import *
#from cython_loss import *

logger = logging.getLogger('ErNESTO-DT')


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
                 random_init: bool = False,
                 temperature_loss=False,
                 **kwargs
                 ):
        self._battery_config = battery_config
        
        self._alg = algorithm
        self._alpha = alpha
        self._beta = beta
        self._n_guesses = n_guesses    
        self._n_jobs = n_jobs
        self._random_init = random_init
        self._noise_SNR = None
        
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
        if 'workers' in kwargs:
            self._n_jobs = kwargs['workers']
        if 'noise_SNR' in kwargs:
            self._noise_SNR = kwargs['noise_SNR']
            self.rng = np.random.default_rng(seed=42)

        self.temperature_loss = temperature_loss

    def _set_theta(self, battery: BatteryEnergyStorageSystem, theta: list):
        battery._electrical_model.r0.resistance, battery._electrical_model.rc.resistance, battery._electrical_model.rc.capacity = theta
        
    def add_noise(self, data: list):
        """
        Add noise to the input data according to the specified SNR.
        """
        v = np.array([record["voltage"] for record in data])
        rms = np.sqrt(np.mean(v**2))
        sigma_from_snr = rms / (10**(self._noise_SNR/20))
        v_noisy = v + self.rng.normal(0, sigma_from_snr, size=v.shape)

        # Replace voltages in a copy of the list
        noisy_records = [{**r, "voltage": float(vn)} for r, vn in zip(data, v_noisy)]
        
        return noisy_records

    def estimate_cluster(self, init_state: dict, input_batch: list):
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
        
        if self._noise_SNR is not None:
            input_batch['voltage'] = self.add_noise(input_batch)
        
        args = (input_batch, init_state, self._battery_config, self._scale_factors, self._alpha, self._beta)
        loss = lambda x: scaled_loss(x, *args)

        results = Parallel(n_jobs=self._n_jobs)(delayed(minimize)(
            loss, x0=guess, method=self._alg, bounds=scaled_bounds, options=self._options)
                                                for guess in initial_guesses)
        
        final_params = [(res.x / self._scale_factors).tolist() for res in results]
        return final_params
            
    
    def estimate_new_theta(self, init_state: dict, input_batch: list, centroid: list = None):
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
        scaled_bounds = [(low * s, high * s) for (low, high), s in zip(self._bounds, self._scale_factors)]    
        
        if centroid is not None and not self._random_init:
            # Create random point following a normal centered into the scaled centroid
            initial_guesses = np.random.normal(loc=centroid * self._scale_factors, scale=0.1, size=(self._n_guesses, len(centroid)))
        else:
             # Scale the bounds according to the scale factors and create initial guesses    
            initial_guesses = [np.array([np.random.uniform(b[0], b[1]) for b in scaled_bounds]) 
                               for _ in range(self._n_guesses)]
            
        if self._noise_SNR is not None:
            input_batch = self.add_noise(input_batch)
                
        # Define the loss function with the scaled parameters
        args = (input_batch, init_state, self._battery_config, self._scale_factors, self._alpha, self._beta)
        loss = lambda x: scaled_loss(x, *args)
                
        results = Parallel(n_jobs=self._n_jobs)(delayed(minimize)(
            loss, x0=guess, method=self._alg, bounds=scaled_bounds, options=self._options)
                                                for guess in initial_guesses)

        res = min(results, key=lambda res: res.fun)
        #print(res)
        best_new_params, best_loss = res.x, res.fun

        print("Estimation of new parameters:")
        print("-----------------------------")
        print(f"New theta: {best_new_params / self._scale_factors}, loss: {best_loss}")
        return best_new_params / self._scale_factors