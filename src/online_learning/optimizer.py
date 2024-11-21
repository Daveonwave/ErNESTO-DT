import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize
from src.digital_twin.bess import BatteryEnergyStorageSystem


class Optimizer:
    def __init__(self, 
                 battery: BatteryEnergyStorageSystem,
                 alg: str,
                 alpha: float,
                 batch_size: int,
                 n_restarts: int,
                 bounds: dict, 
                 scale_factors: dict, 
                 options: dict = None, 
                 temperature_loss=False
                 ):
        # battery attributes
        # TODO: passa l'inizio della window precedente passata dal simulatore!!
        self._battery = battery
        
        self._alg = alg
        self._alpha = alpha    
        self._batch_size = batch_size
        self._n_restarts = n_restarts
        
        self._bounds = bounds
        self._scale_factors = scale_factors

        # optimization attributes:
        self.initial_guess = None
        self.v_hat = None
        self.t_hat = None
        self._v_real = None
        self._i_real = None
        self._t_real = None
        self.dt = None
        self.loss_history = []
        self.best_loss = None
        self.gradient_history = []
        self.options = options
        self.temperature_loss = temperature_loss

    def get_v_hat(self):
        return self.v_hat

    def get_t_hat(self):
        return self.t_hat

    def get_loss_history(self):
        return self.loss_history

    def get_gradient_history(self):
        return self.gradient_history

    def _set_theta(self, theta):
        self._battery._electrical_model.r0.resistance = theta[0]
        self._battery._electrical_model.rc.resistance = theta[1]
        self._battery._electrical_model.rc.capacity = theta[2]

    def lhs(self):
        n = 1
        d = len(self.bounds)
        samples = np.zeros((n, d))
        for i in range(d):
            samples[:, i] = np.random.uniform(low=self.bounds[i][0], high=self.bounds[i][1], size=n)
        for i in range(d):
            np.random.shuffle(samples[:, i])
        return samples.flatten()

    def _battery_load(self, theta):
        # filtro ro, rc, c
        # tenere i,v,t,soc !!!
        # keys = list['key']
        # dict = {key: keys[key] for key in keys}
        self._battery.reset()
        self._battery.init(self.init_info)

        #scaled_theta = [theta[0] * self.scale_factor[0], theta[1] * self.scale_factor[1], theta[2] * self.scale_factor[2]]
        #self._set_theta(scaled_theta)
        self._set_theta(theta)

        elapsed_time = 0
        for k, load in enumerate(self._i_real):
            self._battery.step(load=load, dt=self.dt, k=k)
            self._battery.t_series.append(elapsed_time)
            elapsed_time += self.dt

    def _loss_function(self, theta):
        self._battery_load(theta)

        self.v_hat = self._battery._electrical_model.get_v_series()
        self.v_hat = self.v_hat[1: len(self._v_real) + 1]
        voltage_diff = self.v_hat - self._v_real
        voltage_loss = np.sum(voltage_diff ** 2)

        temperature_loss = 0
        if self.temperature_loss:
           self.t_hat = self._battery._thermal_model.get_temp_series()
           self.t_hat = self.t_hat[1: len(self._t_real) + 1]  # Adjusting for alignment
           temperature_diff = self.t_hat - self._t_real
           temperature_loss = np.sum(temperature_diff ** 2)

        regularization = self.alpha * np.linalg.norm(theta)

        loss = voltage_loss + temperature_loss + regularization

        if loss < self.best_loss:
            self.best_loss = loss
            # self.loss_history.append(self.best_loss)

        return loss

    def optimize_function(self, initial_guess, optimizer_method):
        result = minimize(self._loss_function, initial_guess,
                          method=optimizer_method,
                          bounds=None,  # before was
                          options=self.options)
        return result


    def step(self, i_real, v_real, t_real, alpha, optimizer_method, dt, number_of_restarts, starting_theta, init_info):
        self._i_real = i_real
        self._v_real = v_real
        self._t_real = t_real
        self.dt = dt
        self.alpha = alpha
        self.number_of_restarts = number_of_restarts
        self.best_loss = float('inf')
        self.loss_history = []
        # todo: understand how to retrieve right info from battery_status passed as init_info
        if init_info is not None:
            self.init_info = init_info

        best_value = float('inf')
        best_result = None

        # initial_guess = self.lhs()
        # initial_guess = starting_theta
        initial_guesses = []
        initial_guesses.append(starting_theta)
        #  for ii in range(self.number_of_restarts):
        #  initial_guesses.append(np.array([np.random.uniform(low, high) for low, high in self.bounds]))
        #    initial_guesses.append(self.lhs())
        #  print("initial_guesses are:", initial_guesses)

        results = Parallel(n_jobs=1)(
            delayed(self.optimize_function)(initial_guess, optimizer_method)
            for initial_guess in initial_guesses
        )

        print(type(results))
        print(results)

        #  result = minimize(self._loss_function, initial_guess,
        #                  method=optimizer_method, bounds=self.bounds,
        #                  options=self.options)

        for result in results:
            if result.fun < best_value:
               best_value = result.fun
               best_result = result


        # self.loss_history.append(self.best_loss)

        #return {'r0': best_result.x[0] * self.scale_factor[0],
        #        'rc': best_result.x[1] * self.scale_factor[1],
        #        'c': best_result.x[2] * self.scale_factor[2]}
        return {'r0': best_result.x[0],
                'rc': best_result.x[1],
                'c': best_result.x[2]
                }