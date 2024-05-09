import numpy as np
from scipy.optimize import minimize
from src.digital_twin.bess import BatteryEnergyStorageSystem


class Optimizer:
    def __init__(self, models_config, battery_options, load_var):
        self._v_real = None
        self._i_real = None
        self._temp_battery = None
        # the left value of the capacitor should prevent the nan value
        self.bounds = [(0.001, 0.01), (0.001, 0.01), (0.1, 50000.0)]
        self.initial_guess = None
        self.number_of_restarts = 2
        self.best_results_table = None
        self.dt = 1

        self._models_config = models_config
        self._battery_options = battery_options
        self._load_var = load_var

    def get_status(self):
        return self.best_results_table

    def _equation(self, params):
        theta = params

        # I Should set theta of the thevenin
        self._temp_battery._electrical_model.r0.resistance = theta[0]
        self._temp_battery._electrical_model.rc.resistance = theta[1]
        self._temp_battery._electrical_model.rc.capacity = theta[2]

        #print(theta[0], self._temp_battery._electrical_model.r0.resistance)

        elapsed_time = 0
        #dt = 1

        for k, load in enumerate(self._i_real):
            self._temp_battery.step(load=load, dt=self.dt, k=k)
            self._temp_battery.t_series.append(elapsed_time)
            elapsed_time += self.dt
            #dt = df['time'].iloc[k] - df['time'].iloc[k - 1] if k > 0 else 1.0

        # k differs from len(self.v_real) from 1, one step more always
        rhs = self._temp_battery._electrical_model.get_v_series(k=len(self._v_real))

        diff = rhs - self._v_real

        alpha = 0.1
        loss = np.sum(diff ** 2) + alpha * np.linalg.norm(theta)
        return loss

    def _callback(self, xk):
        loss = self._equation(xk)
        print("Current loss value:", loss)
        print("the theta passed to equation", xk)


    def step(self, i_real, v_real, init_info, dt):
        self._i_real = i_real
        self._v_real = v_real
        self.dt = dt

        results = []
        loss_series = []

        self._temp_battery = BatteryEnergyStorageSystem(models_config=self._models_config,
                                                        battery_options=self._battery_options,
                                                        input_var=self._load_var)

        # Perform multiple restarts
        for _ in range(self.number_of_restarts):
            initial_guess = np.array([np.random.uniform(low, high) for low, high in self.bounds])

            # TODO: CHECK !
            self._temp_battery.reset()
            self._temp_battery.init(init_info=init_info)

            result = minimize(self._equation, initial_guess, method='Nelder-Mead', bounds=self.bounds,
                              callback=self._callback)
            loss_series.append(result.fun)
            if result.fun <= min(loss_series):
                self.best_results_table = self._temp_battery.build_results_table()

            results.append(result)

        best_result = min(results, key=lambda x: x.fun)

        del self._temp_battery

        return {'r0':best_result.x[0], 'rc': best_result.x[1], 'c': best_result.x[2]}