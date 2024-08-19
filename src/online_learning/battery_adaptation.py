from src.digital_twin.bess import BatteryEnergyStorageSystem
from src.online_learning.soc_temp_combination import SocTemp
from src.online_learning.optimizer import Optimizer
from src.online_learning.change_detection.cluster_estimation import cluster_estimation
from src.online_learning.utils import save_to_csv, convert_to_dict_list, save_dict_list_to_csv
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
        self.history_theta = {0: list(), 1: list(), 2: list(), 3: list()}
        self.outliers_sets = {0: list(), 1: list(), 2: list(), 3: list()}
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
            print("loop number:", k)
            elapsed_time += dt
            battery.t_series.append(elapsed_time)
            dt = self.dataset['time'].iloc[k] - self.dataset['time'].iloc[k - 1] if k > 0 else 1.0

            if soc_temp.is_changed(soc, temp):
                print("the combination soc-temperature is changed, current is:", soc_temp.current)

            if (k % self.batch_size == 0 and k != 0):
                print("goin' to optimize:")
                theta = optimizer.step(i_real=self.dataset['i_real'][start:k],
                                       v_real=self.dataset['v_real'][start:k],
                                       t_real=self.dataset['t_real'][start:k],
                                       optimizer_method=self.optimizer_method,
                                       alpha=self.alpha, dt=dt,
                                       number_of_restarts=self.number_of_restarts)
                print("theta is:", theta)
                start = k
                theta_values = np.array(list(theta.values()), dtype=float)
                self.history_theta[soc_temp.current].append(theta_values)
                print("The len of history_theta[current]: ", self.history_theta[soc_temp.current])
                self.v_optimizer = self.v_optimizer + optimizer.get_v_hat()
                # self.temp_optimizer = self.temp_optimizer + optimizer.get_t_hat()

                if self.nominal_clusters[soc_temp.current].contains(theta_values):
                    self.nominal_clusters[soc_temp.current].update(theta_values)

                else:
                    self.outliers_sets[soc_temp.current].append(theta_values)

                    self.phi_hat = cluster_estimation(
                            cluster_data_points=self.nominal_clusters[soc_temp.current].data_points,
                            outliers=self.outliers_sets[soc_temp.current])

                    if self.phi_hat is not None:
                        self.nominal_clusters[soc_temp.current] = self.phi_hat

                # battery._electrical_model.r0.resistance = self.nominal_clusters[soc_temp.current].centroid[0]
                # battery._electrical_model.rc.resistance = self.nominal_clusters[soc_temp.current].centroid[1]
                # battery._electrical_model.rc.capacity = self.nominal_clusters[soc_temp.current].centroid[2]

            battery.step(load, dt, k)
            if len(battery.soc_series) > 1:
                soc = np.mean(np.array(battery.soc_series))
            else:
                soc = battery.soc_series[-1]
            if len(self.dataset['t_real'][start:k]) > 1:
                temp = np.mean(self.dataset['t_real'][start:k], axis=0)
            else:
                temp = battery._thermal_model.get_temp_series(-1)

        if self.save_results:
            print("start savings")
            save_to_csv(self.v_optimizer, 'v_optimizer.csv', ['v_optimizer']) # ok!

            history_theta_0 = convert_to_dict_list(self.history_theta[0])
            save_dict_list_to_csv(dict_list=history_theta_0, filename='phi_0_history_theta.csv')

            history_theta_1 = convert_to_dict_list(self.history_theta[1])
            save_dict_list_to_csv(dict_list=history_theta_1, filename='phi_1_history_theta.csv')

            history_theta_2 = convert_to_dict_list(self.history_theta[2])
            save_dict_list_to_csv(dict_list=history_theta_2, filename='phi_2_history_theta.csv')

            history_theta_3 = convert_to_dict_list(self.history_theta[3])
            save_dict_list_to_csv(dict_list=history_theta_3, filename='phi_3_history_theta.csv')

            # save_to_csv(self.history_theta[0], 'phi_0_history_theta.csv', ['history_theta'])
            # save_to_csv(self.history_theta[1], 'phi_1_history_theta.csv', ['history_theta'])
            # save_to_csv(self.history_theta[2], 'phi_2_history_theta.csv', ['history_theta'])
            # save_to_csv(self.history_theta[3], 'phi_3_history_theta.csv', ['history_theta'])

            outliers_sets_0 = convert_to_dict_list(self.outliers_sets[0])
            save_dict_list_to_csv(dict_list=outliers_sets_0, filename='phi_0_outliers_sets.csv')

            outliers_sets_1 = convert_to_dict_list(self.outliers_sets[1])
            save_dict_list_to_csv(dict_list=outliers_sets_1, filename='phi_1_outliers_sets.csv')

            outliers_sets_2 = convert_to_dict_list(self.outliers_sets[2])
            save_dict_list_to_csv(dict_list=outliers_sets_2, filename='phi_2_outliers_sets.csv')

            outliers_sets_3 = convert_to_dict_list(self.outliers_sets[3])
            save_dict_list_to_csv(dict_list=outliers_sets_3, filename='phi_3_outliers_sets.csv')

            # save_to_csv(self.outliers_sets[0], 'phi_0_outliers_sets.csv', ['Outliers'])
            # save_to_csv(self.outliers_sets[1], 'phi_1_outliers_sets.csv', ['Outliers'])
            # save_to_csv(self.outliers_sets[2], 'phi_2_outliers_sets.csv', ['Outliers'])
            # save_to_csv(self.outliers_sets[3], 'phi_3_outliers_sets.csv', ['Outliers'])

            nominal_cluster_0 = convert_to_dict_list(self.nominal_clusters[0].data_points)
            save_dict_list_to_csv(dict_list=nominal_cluster_0, filename='phi_0.csv')

            nominal_cluster_1 = convert_to_dict_list(self.nominal_clusters[1].data_points)
            save_dict_list_to_csv(dict_list=nominal_cluster_1, filename='phi_1.csv')

            nominal_cluster_2 = convert_to_dict_list(self.nominal_clusters[2].data_points)
            save_dict_list_to_csv(dict_list=nominal_cluster_2, filename='phi_2.csv')

            nominal_cluster_3 = convert_to_dict_list(self.nominal_clusters[3].data_points)
            save_dict_list_to_csv(dict_list=nominal_cluster_3, filename='phi_3.csv')

            # save_to_csv(self.nominal_clusters[0].data_points, 'phi_0.csv', ['phi_0'])
            # save_to_csv(self.nominal_clusters[1].data_points, 'phi_1.csv', ['phi_1'])
            # save_to_csv(self.nominal_clusters[2].data_points, 'phi_2.csv', ['phi_2'])
            # save_to_csv(self.nominal_clusters[3].data_points, 'phi_3.csv', ['phi_3'])

            # todo: solve since it is useful
            # save_to_csv(self.temp_optimizer, 'temp_optimizer.csv', ['temp_optimizer'])
