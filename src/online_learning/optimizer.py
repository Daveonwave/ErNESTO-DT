import numpy as np
from scipy.optimize import minimize
from src.digital_twin.bess import BatteryEnergyStorageSystem

class Optimizer():
    def __init__(self, models_config, battery_options, load_var, init_info):
        # battery attributes
        # TODO: passa l'inizio della window precedente passata dal simulatore!!
        self._temp_battery = BatteryEnergyStorageSystem(models_config=models_config,
                                                        battery_options=battery_options,
                                                        input_var=load_var)
        self.init_info = init_info
        self._temp_battery.init(init_info)
        self._temp_battery.reset()
        # optimization attributes
        self.options = {
                'maxiter': 1000,  # Maximum number of iterations
                'xatol': 1e-12,  # Absolute error in xopt between iterations that is acceptable for convergence
                'fatol': 1e-12,  # Absolute error in func(xopt) between iterations that is acceptable for convergence
                'initial_simplex': None  # Optional initial simplex
            }
        self.bounds = [(0.01, 100), (0.01, 100), (0.01, 100)]
        self.scale_factor = [0.1, 0.1, 1000]
        self.initial_guess = None
        self.number_of_restarts = None
        self.alpha = None
        self.v_hat = None
        self.t_hat = None
        self._v_real = None
        self._i_real = None
        self.dt = None
        self.loss_history = []

    def get_v_hat(self):
        return self.v_hat

    def get_t_hat(self):
        return self.t_hat

    def _set_theta(self, theta):
        self._temp_battery._electrical_model.r0.resistance = theta[0]
        self._temp_battery._electrical_model.rc.resistance = theta[1]
        self._temp_battery._electrical_model.rc.capacity = theta[2]


    def _battery_load(self, theta):
        self._temp_battery.reset()
        # filtro ro, rc, c
        # tenere i,v,t,soc !!!
        # keys = list['key']
        # dict = {key: keys[key] for key in keys}

        self._temp_battery.init(self.init_info)

        scaled_theta = [theta[0] * self.scale_factor[0], theta[1] * self.scale_factor[1], theta[2] * self.scale_factor[2]]

        self._set_theta(scaled_theta)

        elapsed_time = 0
        for k, load in enumerate(self._i_real):
            self._temp_battery.step(load=load, dt=self.dt, k=k)
            self._temp_battery.t_series.append(elapsed_time)
            elapsed_time += self.dt

    def _callback(self, xk):
        print("LOSS VALUE:", self.loss_history[-1])
        print("THETA", xk)

    def _loss_function(self, theta):
        self._battery_load(theta)

        self.v_hat = self._temp_battery._electrical_model.get_v_series()
        self.v_hat = self.v_hat[1: len(self._v_real) + 1]

        diff = self.v_hat - self._v_real
        sum = np.sum(diff ** 2)
        regularization = self.alpha * np.linalg.norm(theta)
        loss = sum + regularization

        self.loss_history.append(loss)

        self.t_hat = self._temp_battery._thermal_model.get_temp_series()

        return loss

    def step(self, i_real, v_real, alpha, optimizer_method, dt, number_of_restarts, ):
        self._i_real = i_real
        self._v_real = v_real
        self.dt = dt
        self.alpha = alpha
        self.number_of_restarts = number_of_restarts

        self.loss_history = []

        result = None

        #TODO: IMPLEMENT MULTIPLE RESTARTS:

        for _ in range(self.number_of_restarts):
            initial_guess = np.array([np.random.uniform(low, high) for low, high in self.bounds])

            result = minimize(self._loss_function, initial_guess, method=optimizer_method, bounds=self.bounds,
                              options=self.options, callback= self._callback) #

        return {'r0':result.x[0] * self.scale_factor[0], 'rc':result.x[1]* self.scale_factor[1], 'c':result.x[2]* self.scale_factor[2]}

