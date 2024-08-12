from src.digital_twin.bess import BatteryEnergyStorageSystem
from src.online_learning.soc_temp_combination import SocTemp
from src.online_learning.optimizer import Optimizer
from src.online_learning.change_detection.cluster_estimation import cluster_estimation
import numpy as np


class BatteryAdaptation:
    def __init__(self, optimizer_settings, battery_setings, dataset, nominal_clusters):
        # optimizer:
        self.alpha = optimizer_settings['alpha']
        self.batch_size = optimizer_settings['batch_size']
        self.optimizer_method = optimizer_settings['optimizer_method']
        self.save_results = optimizer_settings['save_results']
        self.number_of_restarts = optimizer_settings['number_of_restarts']
        self.bounds = optimizer_settings['bounds']
        self.scale_factors = optimizer_settings['scale_factors']
        # battery:
        self.ranges = battery_setings['ranges']
        self.electrical_params = battery_setings['electrical_params']
        self.thermal_params = battery_setings['thermal_params']
        self.models_config = [self.electrical_params, self.thermal_params]
        self.battery_options = battery_setings['battery_options']
        self.load_var = battery_setings['load_var']
        # inputs for the alg.
        self.dataset = dataset
        self.nominal_clusters = nominal_clusters  # check coherency with the id
        # data structures for the alg.
        self.history_theta = list()
        self.outliers_sets = {'0': None, '1': None, '2': None, '3': None}
        self.v_optimizer = list()
        self.temp_optimizer = list()
        self.phi_hat = None

    def run_experiment(self):

        battery = BatteryEnergyStorageSystem(
            models_config=self.models_config,
            battery_options=self.battery_options,
            input_var=self.load_var
        )
        battery.reset(reset_info={'electricala_params': self.electrical_params,
                                  'thermal_params': self.thermal_params})
        battery.init({'dissipated_heat': 0})  # check if you can remove it

        optimizer = Optimizer(models_config=self.models_config, battery_options=self.battery_options,
                              load_var=self.load_var,
                              init_info={'electricala_params': self.electrical_params,
                                         'thermal_params': self.thermal_params},
                              bounds=self.bounds, scale_factor=self.scale_factors,
                              options=None)

        soc = battery.soc_series[-1]
        temp = battery._thermal_model.get_temp_series(-1)
        soc_temp = SocTemp(self.ranges, soc, temp)

        elapsed_time = 0
        dt = 1
        start = 0
        for k, load in enumerate(self.dataset['i_real']):
            elapsed_time += dt
            battery.t_series.append(elapsed_time)
            dt = self.dataset['time'].iloc[k] - self.dataset['time'].iloc[k - 1] if k > 0 else 1.0

            if (k % self.batch_size == 0 and k != 0) or soc_temp.is_changed(soc, temp):

                theta = optimizer.step(i_real=self.dataset['i_real'][start:k],
                                       v_real=self.dataset['v_real'][start:k],
                                       t_real=self.dataset['t_real'][start:k],
                                       optimizer_method=self.optimizer_method,
                                       alpha=self.alpha, dt=dt,
                                       number_of_restarts=self.number_of_restarts)

                start = k
                self.history_theta.append(theta)
                self.v_optimizer = self.v_optimizer + optimizer.get_v_hat()
                self.temp_optimizer = self.temp_optimizer + optimizer.get_t_hat()

                if self.nominal_clusters[soc_temp.current].contains(np.array(list(theta))):
                    self.nominal_clusters[soc_temp.current].update()

                else:
                    self.outliers_sets[soc_temp.current].append(theta)

                    self.phi_hat = cluster_estimation(
                            cluster_data_points=self.nominal_clusters[soc_temp.current].data_points,
                            outliers=self.outliers_sets[soc_temp.current])

                    if self.phi_hat is not None:
                        self.nominal_clusters[soc_temp.current] = self.phi_hat

                battery._electrical_model.r0.resistance = self.nominal_clusters[soc_temp.current].centroid[0]
                battery._electrical_model.rc.resistance = self.nominal_clusters[soc_temp.current].centroid[1]
                battery._electrical_model.rc.capacity = self.nominal_clusters[soc_temp.current].centroid[2]

            battery.step(load, dt, k)
            soc = battery.soc_series[-1]
            temp = np.mean(self.dataset['temperature'].value, axis=0)  # don't fall on the edge

        if self.save_results:
            pass
