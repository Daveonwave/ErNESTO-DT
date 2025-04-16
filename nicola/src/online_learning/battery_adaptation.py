from src.digital_twin.bess import BatteryEnergyStorageSystem
from src.online_learning.soc_temp_combination_kmeans import SocTemp
from src.online_learning.optimizer import Optimizer
from src.online_learning.change_detection.cluster_estimation import cluster_estimation
from src.online_learning.utils import save_to_csv, convert_to_dict_list, save_dict_list_to_csv
from tqdm import tqdm
import numpy as np
import pickle



class BatteryAdaptation:
    def __init__(self, optimizer_settings, battery_settings, dataset, nominal_clusters):
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
        self.ranges = battery_settings['ranges']
        self.electrical_params = battery_settings['electrical_params']
        self.thermal_params = battery_settings['thermal_params']
        self.models_config = [self.electrical_params, self.thermal_params]
        self.battery_options = battery_settings['battery_options']
        self.load_var = battery_settings['load_var']
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
        #battery.init({'dissipated_heat': 0})  # check if you can remove it

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

        elapsed_time = 0
        dt = 1
        start = 0
        for k, load in tqdm(enumerate(self.dataset['i_real'])):
            print("loop number:", k)
            elapsed_time += dt
            battery.t_series.append(elapsed_time)
            dt = self.dataset['time'].iloc[k] - self.dataset['time'].iloc[k - 1] if k > 0 else 1.0

            if (k % self.batch_size == 0 and k != 0):
                print("______________________________________________________________")
                print("the result of get status table:", battery.get_status_table())
                print("______________________________________________________________")
                print("goin' to optimize:")
                # todo: find a better way of initialize !!!!!
                initial_guess = np.array([np.random.uniform(low, high) for low, high in self.bounds])
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
                actual_digital_twin_theta = {'r0': r0, 'rc': rc, 'c': c}
                print("______________________________________________________________")
                print("theta of the dt, i.e. the previous one is:", actual_digital_twin_theta)
                print("theta_hat is:", theta)
                print("______________________________________________________________")

                start = k
                theta_values = np.array(list(theta.values()), dtype=float)
                self.history_theta[soc_temp.current].append(theta_values)  # you can store also the dict version

                self.v_optimizer = self.v_optimizer + optimizer.get_v_hat()
                # self.temp_optimizer = self.temp_optimizer + optimizer.get_t_hat()

                if self.nominal_clusters[soc_temp.current].contains(theta_values):
                    pass  # discussed with prof. maybe it could bring to problems
                    # self.nominal_clusters[soc_temp.current].update(theta_values)

                else:
                    self.outliers_sets[soc_temp.current].append(theta_values)

                    self.phi_hat = cluster_estimation(
                            cluster_data_points=self.nominal_clusters[soc_temp.current].data_points,
                            outliers=self.outliers_sets[soc_temp.current])

                    if self.phi_hat is not None:
                        with open(f"phi_hat_{soc_temp.current}.pkl", "wb") as f:
                            pickle.dump(self.phi_hat, f)

                        self.nominal_clusters[soc_temp.current] = self.phi_hat
                        self.nominal_clusters[soc_temp.current].compute_centroid()
                        self.nominal_clusters[soc_temp.current].compute_covariance()

                (battery._electrical_model.r0.resistance,
                 battery._electrical_model.rc.resistance,
                 battery._electrical_model.rc.capacity) = self.nominal_clusters[soc_temp.current].centroid

            battery.step(load, dt, k)

            soc = battery.soc_series[-1]  # The mean gave me problems

            if len(self.dataset['t_real'][start:k]) > 1:
                temp = np.mean(self.dataset['t_real'][start:k], axis=0)
            else:
                temp = battery._thermal_model.get_temp_series(-1)

            soc_temp.check_cluster(temp=temp, soc=soc)
            battery_status = battery.get_status_table()

        if self.save_results:
            print("start savings")
            save_to_csv(self.v_optimizer, 'v_optimizer.csv', ['v_optimizer'])  # ok!
            # here I was saving the list of thetas and socs, useful ???

            # history of thetas for each cluster
            history_theta_0 = convert_to_dict_list(self.history_theta[0])
            save_dict_list_to_csv(dict_list=history_theta_0, filename='phi_0_history_theta.csv')

            history_theta_1 = convert_to_dict_list(self.history_theta[1])
            save_dict_list_to_csv(dict_list=history_theta_1, filename='phi_1_history_theta.csv')

            history_theta_2 = convert_to_dict_list(self.history_theta[2])
            save_dict_list_to_csv(dict_list=history_theta_2, filename='phi_2_history_theta.csv')

            history_theta_3 = convert_to_dict_list(self.history_theta[3])
            save_dict_list_to_csv(dict_list=history_theta_3, filename='phi_3_history_theta.csv')

            # outliers for each cluster
            outliers_sets_0 = convert_to_dict_list(self.outliers_sets[0])
            save_dict_list_to_csv(dict_list=outliers_sets_0, filename='phi_0_outliers_sets.csv')

            outliers_sets_1 = convert_to_dict_list(self.outliers_sets[1])
            save_dict_list_to_csv(dict_list=outliers_sets_1, filename='phi_1_outliers_sets.csv')

            outliers_sets_2 = convert_to_dict_list(self.outliers_sets[2])
            save_dict_list_to_csv(dict_list=outliers_sets_2, filename='phi_2_outliers_sets.csv')

            outliers_sets_3 = convert_to_dict_list(self.outliers_sets[3])
            save_dict_list_to_csv(dict_list=outliers_sets_3, filename='phi_3_outliers_sets.csv')

            # new nominal clusters
            nominal_cluster_0 = convert_to_dict_list(self.nominal_clusters[0].data_points)
            save_dict_list_to_csv(dict_list=nominal_cluster_0, filename='phi_0.csv')

            nominal_cluster_1 = convert_to_dict_list(self.nominal_clusters[1].data_points)
            save_dict_list_to_csv(dict_list=nominal_cluster_1, filename='phi_1.csv')

            nominal_cluster_2 = convert_to_dict_list(self.nominal_clusters[2].data_points)
            save_dict_list_to_csv(dict_list=nominal_cluster_2, filename='phi_2.csv')

            nominal_cluster_3 = convert_to_dict_list(self.nominal_clusters[3].data_points)
            save_dict_list_to_csv(dict_list=nominal_cluster_3, filename='phi_3.csv')
