import numpy as np
from scipy.optimize import minimize
from src.digital_twin.bess import BatteryEnergyStorageSystem

class Optimizer():
    def __init__(self, models_config, battery_options, load_var, init_info,
                 bounds, scale_factor,options , temperature_loss=False):
        # battery attributes
        # TODO: passa l'inizio della window precedente passata dal simulatore!!
        self._temp_battery = BatteryEnergyStorageSystem(models_config=models_config,
                                                        battery_options=battery_options,
                                                        input_var=load_var)
        self.init_info = init_info
        self._temp_battery.init(init_info)
        self._temp_battery.reset()
        self.bounds = bounds
        self.scale_factor = scale_factor

        # optimization attributes:
        self.initial_guess = None
        self.number_of_restarts = None
        self.alpha = None
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
        self._temp_battery._electrical_model.r0.resistance = theta[0]
        self._temp_battery._electrical_model.rc.resistance = theta[1]
        self._temp_battery._electrical_model.rc.capacity = theta[2]

    def lhs(self):

        n = 1
        d = len(self.bounds)
        samples = np.zeros((n, d))

        for i in range(d):
            samples[:, i] = np.random.uniform(low=self.bounds[i][0], high=self.bounds[i][1], size=n)

        for i in range(d):
            np.random.shuffle(samples[:, i])

        return samples.flatten()

    def finite_diff(self, x):
        h = 1e-5  # Step size for finite differences
        gradients = []
        for i in range(len(x)):
            x_plus_h = np.copy(x)
            x_plus_h[i] += h
            loss_plus_h = self._loss_function(x_plus_h)
            gradient_i = (loss_plus_h - self.best_loss) / h
            gradients.append(gradient_i)
        return gradients

    def _battery_load(self, theta):
        # filtro ro, rc, c
        # tenere i,v,t,soc !!!
        # keys = list['key']
        # dict = {key: keys[key] for key in keys}
        self._temp_battery.reset()
        self._temp_battery.init(self.init_info)

        scaled_theta = [theta[0] * self.scale_factor[0], theta[1] * self.scale_factor[1], theta[2] * self.scale_factor[2]]
        self._set_theta(scaled_theta)

        elapsed_time = 0
        for k, load in enumerate(self._i_real):
            self._temp_battery.step(load=load, dt=self.dt, k=k)
            self._temp_battery.t_series.append(elapsed_time)
            elapsed_time += self.dt

    def _callback(self, xk):
        print("THETA", xk)
        gradient = self.finite_diff(xk)
        self.gradient_history.append(gradient)
        print("the gradient is :", gradient)
        #print("LOSS VALUE:", self.loss_history[-1])

    def _loss_function(self, theta):
        self._battery_load(theta)

        self.v_hat = self._temp_battery._electrical_model.get_v_series()
        self.v_hat = self.v_hat[1: len(self._v_real) + 1]
        voltage_diff = self.v_hat - self._v_real
        voltage_loss = np.sum(voltage_diff ** 2)

        temperature_loss = 0
        if self.temperature_loss:
           self.t_hat = self._temp_battery._thermal_model.get_temp_series()
           self.t_hat = self.t_hat[1: len(self._t_real) + 1]  # Adjusting for alignment
           temperature_diff = self.t_hat - self._t_real
           temperature_loss = np.sum(temperature_diff ** 2)

        regularization = self.alpha * np.linalg.norm(theta)

        loss = voltage_loss + temperature_loss + regularization

        if loss < self.best_loss:
            self.best_loss = loss
            #self.loss_history.append(self.best_loss)

        return loss

    def step(self, i_real, v_real, t_real, alpha, optimizer_method, dt, number_of_restarts):
        self._i_real = i_real
        self._v_real = v_real
        self._t_real = t_real
        self.dt = dt
        self.alpha = alpha
        self.number_of_restarts = number_of_restarts
        self.best_loss = float('inf')

        self.loss_history = []

        best_value = float('inf')
        best_result = None

        for ii in range(self.number_of_restarts):
            print("restart number :", ii)
            initial_guess = np.array([np.random.uniform(low, high) for low, high in self.bounds])
            #initial_guess = self.lhs()

            result = minimize(self._loss_function, initial_guess,
                              method=optimizer_method, bounds=self.bounds,
                              callback=self._callback, options=self.options)

            if result.fun < best_value:
                best_result = result
                best_value = result.fun

        self.loss_history.append(self.best_loss)

        return {'r0': best_result.x[0] * self.scale_factor[0],
                'rc': best_result.x[1] * self.scale_factor[1],
                'c': best_result.x[2] * self.scale_factor[2]}

