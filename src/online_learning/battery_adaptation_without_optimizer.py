from src.digital_twin.bess import BatteryEnergyStorageSystem
from src.online_learning.soc_temp_combination_kmeans import SocTemp
from src.online_learning.change_detection.surrogate_data.surrogated_optimizer import GenerationMechanism
from src.online_learning.change_detection.cluster_estimation import cluster_estimation
import numpy as np


class BatteryAdaptationWO:
    def __init__(self, battery_setings, dataset, nominal_clusters):
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

        flag = None

        history_temp = {0: list(), 1: list(), 2: list(), 3: list()}
        history_soc = {0: list(), 1: list(), 2: list(), 3: list()}

        battery = BatteryEnergyStorageSystem(
            models_config=self.models_config,
            battery_options=self.battery_options,
            input_var=self.load_var
        )
        battery.reset(reset_info={'electricala_params': self.electrical_params,
                                  'thermal_params': self.thermal_params})
        battery.init({'dissipated_heat': 0})  # check if you can remove it

        # todo: fix
        #optimizer = Optimizer(models_config=self.models_config, battery_options=self.battery_options,
        #                      load_var=self.load_var,
        #                      init_info={'electricala_params': self.electrical_params,
        #                                 'thermal_params': self.thermal_params},
        #                      bounds=self.bounds, scale_factor=self.scale_factors,
        #                      options=self.options)

        # sum of samples must be: 550

        generator = GenerationMechanism(mean=[0.009716, 0.0123, 9867.013],
                                        cov=[[1, 0.01, 0.08], [0.1, 1, 0.04], [0.03, 0.04, 1]],
                                        num_samples=450)
        generator.generate_data()
        generator.generate_outliers(num_outliers=20)
        generator.generate_around_outliers(num_samples_around_outliers=80)
        theta_values_list = generator.results



        soc = battery.soc_series[-1]
        temp = battery._thermal_model.get_temp_series(-1)
        soc_temp = SocTemp(clusters_ranges=self.ranges)
        soc_temp.check_cluster(soc=soc, temp=temp)
        print('the current cluster is:', soc_temp.current)

        elapsed_time = 0
        dt = 1
        start = 0
        for k, load in enumerate(self.dataset['i_real']):
            print("loop number:", k)
            elapsed_time += dt
            battery.t_series.append(elapsed_time)
            dt = self.dataset['time'].iloc[k] - self.dataset['time'].iloc[k - 1] if k > 0 else 1.0

            change_cell_counter = 0

            if (k % 1 == 0 and k != 0):
                print("goin' to optimize:")
                theta_values = theta_values_list[k]
                #print(f"theta_values shape: {np.array(theta_values).shape}")
                #print("theta is:", theta_values)

                start = k
                #theta_values = np.array(list(theta.values()), dtype=float)
                self.history_theta[soc_temp.current].append(theta_values)
                #print("The len of history_theta[current]: ", self.history_theta[soc_temp.current])
                #self.v_optimizer = self.v_optimizer + optimizer.get_v_hat()
                # self.temp_optimizer = self.temp_optimizer + optimizer.get_t_hat()

                if self.nominal_clusters[soc_temp.current].contains(theta_values):
                    print('the point is contained')
                    pass
                    #self.nominal_clusters[soc_temp.current].update(theta_values)
                else:
                    self.outliers_sets[soc_temp.current].append(theta_values)


                    self.phi_hat = cluster_estimation(
                            cluster_data_points=self.nominal_clusters[soc_temp.current].data_points,
                            outliers=self.outliers_sets[soc_temp.current])

                    if self.phi_hat is not None:
                        flag = True
                        self.nominal_clusters[soc_temp.current].data_points = self.phi_hat.data_points.tolist()
                        self.nominal_clusters[soc_temp.current].compute_centroid()
                        self.nominal_clusters[soc_temp.current].compute_covariance()

                (battery._electrical_model.r0.resistance,
                 battery._electrical_model.rc.resistance,
                 battery._electrical_model.rc.capacity) = self.nominal_clusters[soc_temp.current].centroid

            battery.step(load, dt, k)

            soc = battery.soc_series[-1]  # The mean gave me problems
            history_soc[soc_temp.current].append(soc)

            if len(self.dataset['t_real'][start:k]) > 1:
                temp = np.mean(self.dataset['t_real'][start:k], axis=0)
                history_temp[soc_temp.current].append(temp)
            else:
                temp = battery._thermal_model.get_temp_series(-1)
                history_temp[soc_temp.current].append(temp)

            print("the number of changing is:", change_cell_counter)
            if flag:
               print('I have estimated a change detection')
        if True:
            # for testing purpose
            r0 = battery._electrical_model.r0.resistance
            rc = battery._electrical_model.rc.resistance
            c = battery._electrical_model.rc.capacity
            actual_digital_twin_theta = {'r0': r0, 'rc': rc, 'c': c}
            #print("theta of the dt, i.e. the previous one is:", actual_digital_twin_theta)
            print("start savings")

            np.savetxt('new_phi_3.csv', np.vstack(self.nominal_clusters[3].data_points),
                       delimiter=',', header='r0,rc,c', comments='')

            np.savetxt('new_phi_2.csv', np.vstack(self.nominal_clusters[2].data_points),
                       delimiter=',', header='r0,rc,c', comments='')

            np.savetxt('new_phi_1.csv', np.vstack(self.nominal_clusters[1].data_points),
                       delimiter=',', header='r0,rc,c', comments='')

            np.savetxt('new_phi_0.csv', np.vstack(self.nominal_clusters[0].data_points),
                       delimiter=',', header='r0,rc,c', comments='')

