from src.digital_twin.bess import BatteryEnergyStorageSystem
from src.online_learning.soc_temp_combination_kmeans import SocTemp
from src.online_learning.optimizer import Optimizer
from src.online_learning.change_detection.cluster_estimation import cluster_estimation
from src.online_learning.utils import save_to_csv
import numpy as np
import pandas as pd
import pickle



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
        self.options = optimizer_settings['options']
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
        self.outliers_set = list()
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
                              options=self.options)

        soc = battery.soc_series[-1]
        temp = battery._thermal_model.get_temp_series(-1)
        battery_status = battery.get_status_table()
        soc_temp = SocTemp(self.ranges)
        soc_temp.check_cluster(temp=temp, soc=soc)

        elapsed_time = 0
        dt = 1
        start = 0
        for k, load in enumerate(self.dataset['i_real']):
            print("loop number:", k)
            elapsed_time += dt
            battery.t_series.append(elapsed_time)
            dt = self.dataset['time'].iloc[k] - self.dataset['time'].iloc[k - 1] if k > 0 else 1.0

            if (k % self.batch_size == 0 and k != 0):
                print("______________________________________________________________")
                print("the result of get status table:", battery.get_status_table())
                print("______________________________________________________________")
                print("goin' to optimize:")

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

                self.v_optimizer = self.v_optimizer + optimizer.get_v_hat()
                # self.temp_optimizer = self.temp_optimizer + optimizer.get_t_hat()

                if self.nominal_clusters[soc_temp.current].contains(theta_values):
                    pass  # discussed with prof. maybe it could bring to problems
                    # self.nominal_clusters[soc_temp.current].update(theta_values)

                else:
                    self.outliers_set.append(theta)

            battery.step(load, dt, k)

            soc = battery.soc_series[-1]  # The mean gave me problems
            if len(self.dataset['t_real'][start:k]) > 1:
                temp = np.mean(self.dataset['t_real'][start:k], axis=0)
            else:
                temp = battery._thermal_model.get_temp_series(-1)

            soc_temp.check_cluster(temp=temp, soc=soc)
            battery_status = battery.get_status_table()

        self.phi_hat = cluster_estimation(
            # check if it's meaningful to comapre the following two sets
            cluster_data_points=self.nominal_clusters[soc_temp.current].data_points,
            outliers=self.outliers_set)

        if self.phi_hat is not None:
            with open(f"phi_hat_{soc_temp.current}.pkl", "wb") as f:
                pickle.dump(self.phi_hat, f)

        if self.save_results:
            print("start savings")
            save_to_csv(self.v_optimizer, 'v_optimizer.csv', ['v_optimizer'])  # ok!

            df_outliers = pd.DataFrame(self.outliers_set)
            df_outliers.to_csv('outliers.csv', index=False)

            df = pd.DataFrame( np.vstack(self.phi_hat.data_points), columns=['r0', 'rc', 'c'])
            df.to_csv('phi_hat.csv', index=False)