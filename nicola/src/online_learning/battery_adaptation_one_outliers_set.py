from src.digital_twin.bess import BatteryEnergyStorageSystem
from src.online_learning.soc_temp_combination_kmeans import SocTemp
from src.online_learning.optimizer import Optimizer
from src.online_learning.change_detection.cluster_estimation import cluster_estimation
from src.online_learning.utils import save_to_csv, convert_to_dict_list, save_dict_list_to_csv
from datetime import datetime
import os
import numpy as np
import pandas as pd


class BatteryAdaptation:
    def __init__(self, optimizer_settings, battery_setings, change_detection_settings,
                 dataset, nominal_clusters, input_info):
        # optimizer:
        self.alpha = optimizer_settings['alpha']
        self.batch_size = optimizer_settings['batch_size']
        self.optimizer_method = optimizer_settings['optimizer_method']
        self.save_results = optimizer_settings['save_results']
        self.number_of_restarts = optimizer_settings['number_of_restarts']
        self.bounds = optimizer_settings['bounds']
        self.scale_factors = optimizer_settings['scale_factors']
        self.options = optimizer_settings['options']
        # battery:
        self.ranges = battery_setings['ranges']
        self.electrical_params = battery_setings['electrical_params']
        self.thermal_params = battery_setings['thermal_params']
        self.models_config = [self.electrical_params, self.thermal_params]
        self.battery_options = battery_setings['battery_options']
        self.load_var = battery_setings['load_var']
        # inputs for the alg.
        self.input_info = input_info
        self.dataset = dataset
        self.nominal_clusters = nominal_clusters  # check coherency with the id
        # data structures for the alg.
        self.change_cell_history = list()
        self.history_theta_good = {0: list(), 1: list(), 2: list(), 3: list()}
        self.temp_history = list()
        self.soc_history = list()
        self.outliers_set = list()
        self.v_optimizer = list()
        self.temp_optimizer = list()
        self.phi_hat_list = list()
        # change detection hyperparameters
        self.epsilon = change_detection_settings['epsilon']
        self.radius = change_detection_settings['radius']
        self.p = change_detection_settings['p']
        self.minimum_data_points = change_detection_settings['minimum_data_points']
        self.support_fraction = change_detection_settings['support_fraction']

    def run_experiment(self):

        battery = BatteryEnergyStorageSystem(
            models_config=self.models_config,
            battery_options=self.battery_options,
            input_var=self.load_var
        )
        battery.reset()
        battery.init()

        optimizer = Optimizer(models_config=self.models_config, battery_options=self.battery_options,
                              load_var=self.load_var,
                              init_info={'electricala_params': self.electrical_params,
                                         'thermal_params': self.thermal_params},
                              bounds=self.bounds, scale_factor=self.scale_factors,
                              options=self.options)

        soc = battery.soc_series[-1]
        temp = battery._thermal_model.get_temp_series(-1)
        battery_status = battery.get_status_table()
        soc_temp = SocTemp(self.ranges)
        soc_temp.check_cluster(temp=temp, soc=soc)
        current_cell = soc_temp.current

        elapsed_time = 0
        dt = 1
        start = 0
        for k, load in enumerate(self.dataset['i_real']):
            print('loop number:', k)
            elapsed_time += dt
            battery.t_series.append(elapsed_time)
            dt = self.dataset['time'].iloc[k] - self.dataset['time'].iloc[k - 1] if k > 0 else 1.0

            if k % self.batch_size == 0 and k != 0:
                print("______________________________________________________________")
                print("the result of get status table:", battery.get_status_table())
                print("______________________________________________________________")

                self.nominal_clusters[soc_temp.current].compute_centroid()
                initial_guess = self.nominal_clusters[soc_temp.current].centroid
                # initial_guess = np.array([np.random.uniform(low, high) for low, high in self.bounds])

                theta = optimizer.step(i_real=self.dataset['i_real'][start:k],
                                       v_real=self.dataset['v_real'][start:k],
                                       t_real=self.dataset['t_real'][start:k],
                                       optimizer_method=self.optimizer_method,
                                       alpha=self.alpha, dt=dt,
                                       number_of_restarts=self.number_of_restarts,
                                       starting_theta=initial_guess,
                                       init_info=battery_status)

                r0 = battery._electrical_model.r0.resistance
                rc = battery._electrical_model.rc.resistance
                c = battery._electrical_model.rc.capacity

                print("______________________________________________________________")
                print("theta of the dt, i.e. the previous one is:", {'r0': r0, 'rc': rc, 'c': c})
                print("theta_hat is:", theta)
                print("______________________________________________________________")

                start = k
                theta_values = np.array(list(theta.values()), dtype=float)

                self.v_optimizer = self.v_optimizer + optimizer.get_v_hat()
                # self.temp_optimizer = self.temp_optimizer + optimizer.get_t_hat()

                if self.nominal_clusters[soc_temp.current].contains(theta_values):
                    self.history_theta_good[soc_temp.current].append(theta_values)
                    pass  # discussed with prof. maybe it could bring to problems
                    # self.nominal_clusters[soc_temp.current].update(theta_values)

                else:
                    self.outliers_set.append(theta_values)

            (battery._electrical_model.r0.resistance,
             battery._electrical_model.rc.resistance,
             battery._electrical_model.rc.capacity) = self.nominal_clusters[soc_temp.current].centroid

            battery.step(load, dt, k)
            soc = battery.soc_series[-1]  # The mean gave me problems
            self.soc_history.append(soc)

            if len(self.dataset['t_real'][start:k]) > 1:
                temp = np.mean(self.dataset['t_real'][start:k], axis=0)
                self.temp_history.append(temp)
            else:
                temp = battery._thermal_model.get_temp_series(-1)  # it should take the first of the real dataset
                self.temp_history.append(temp)

            soc_temp.check_cluster(temp=temp, soc=soc)
            next_cell = soc_temp.current
            if next_cell != current_cell:
                self.change_cell_history.append(k)
                current_cell = next_cell

            battery_status = battery.get_status_table()

        if len(self.outliers_set) >= self.minimum_data_points:
            self.phi_hat_list = cluster_estimation(
                # check if it's meaningful to comapare the following two sets
                cluster_data_points=self.nominal_clusters[soc_temp.current].data_points,
                outliers=self.outliers_set, epsilon=self.epsilon, radius=self.radius,
                p=self.p, minimum_data_points=self.minimum_data_points, support_fraction=self.support_fraction)

        if self.save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(timestamp, exist_ok=True)

            # v_hat
            print("saving the v_hat from the optimizer")
            save_to_csv(self.v_optimizer, f'{timestamp}/v_optimizer.csv', ['v_optimizer'])  # ok!

            # outliers
            print("saving the outliers")
            outliers_set = convert_to_dict_list(self.outliers_set)
            save_dict_list_to_csv(dict_list=outliers_set, filename=f'{timestamp}/outliers_set.csv')

            # thetas that are contained whitin the clusters
            print("saving the history theta good for each cluster")
            history_theta_good_0 = convert_to_dict_list(self.history_theta_good[0])
            save_dict_list_to_csv(dict_list=history_theta_good_0, filename=f'{timestamp}/history_theta_good_0.csv')

            history_theta_good_1 = convert_to_dict_list(self.history_theta_good[1])
            save_dict_list_to_csv(dict_list=history_theta_good_1, filename=f'{timestamp}/history_theta_good_1.csv')

            history_theta_good_2 = convert_to_dict_list(self.history_theta_good[2])
            save_dict_list_to_csv(dict_list=history_theta_good_2, filename=f'{timestamp}/history_theta_good_2.csv')

            history_theta_good_3 = convert_to_dict_list(self.history_theta_good[3])
            save_dict_list_to_csv(dict_list=history_theta_good_3, filename=f'{timestamp}/history_theta_good_3.csv')

            # faulty clusters
            print("saving the faulty clusters")
            for i, phi_hat in enumerate(self.phi_hat_list):
                if phi_hat is not None:
                    faulty = convert_to_dict_list(phi_hat.data_points)
                    save_dict_list_to_csv(dict_list=faulty, filename=f'{timestamp}/phi_{i}.csv')

            # temp series
            print("saving the soc,temp and change cell series")
            df_temp_history = pd.DataFrame({'temperature': self.temp_history})
            df_temp_history.to_csv(f'{timestamp}/temp_history.csv', index=False)

            # soc series
            df_soc_history = pd.DataFrame({'soc': self.soc_history})
            df_soc_history.to_csv(f'{timestamp}/soc_history.csv', index=False)

            # k_step of change cell series
            df_k_step_series = pd.DataFrame({'k_step': self.change_cell_history})
            df_k_step_series.to_csv(f'{timestamp}/change_cell_history.csv', index=False)

            print("saving settings and info")
            # optimizer hyperparams
            with open(f'{timestamp}/info_and_settings.txt', 'w') as file:
                file.write('optimizer settings: \n')
                file.write(f'alpha: {self.alpha}\n')
                file.write(f'batchsize: {self.batch_size}\n')
                file.write(f'optimizer_method: {self.optimizer_method}\n')
                file.write(f'save_results: {self.save_results}\n')
                file.write(f'number_of_restarts: {self.number_of_restarts}\n')
                file.write(f'options: {self.options}\n')
                file.write(f'bounds: {self.bounds}\n')
                file.write('\n')
                file.write('change detection settings: \n')
                file.write(f'epsilon: {self.epsilon}\n')
                file.write(f'radius: {self.radius}\n')
                file.write(f'p: {self.p}\n')
                file.write(f'minimum_data_points: {self.minimum_data_points}\n')
                file.write(f'support_fraction: {self.support_fraction}\n')
                file.write('\n')
                file.write('input_info: \n')
                for i, elem in enumerate(self.input_info):
                    file.write(f'{i}: {elem}\n')
