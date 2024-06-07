from src.digital_twin.bess import BatteryEnergyStorageSystem
from src.online_learning.cluster import Cluster
from src.online_learning.grid import Grid
from src.online_learning.optimizer import Optimizer
from src.online_learning.adaptation import fault_cluster_creation
import pandas as pd
import numpy as np
import yaml
import matplotlib.pyplot as plt

class Simulation:
    def __init__(self, alpha, batch_size, optimizer_method, training_window, save_results, number_of_restarts):
        self.alpha = alpha
        self.batch_size = batch_size
        self.optimizer_method = optimizer_method
        self.training_window = training_window
        self.save_results = save_results
        self.number_of_restarts = number_of_restarts

    def load_electrical_params_from_yaml(self, file_path):
        with open(file_path, 'r') as file:
            electrical_params = yaml.safe_load(file)
        return electrical_params

    def load_thermal_params_from_yaml(self, file_path):
        with open(file_path, 'r') as file:
            thermal_params = yaml.safe_load(file)
        return thermal_params

    def load_battery_options_from_yaml(self, file_path):
        with open(file_path, 'r') as file:
            battery_options = yaml.safe_load(file)
        return battery_options

    def load_grid_parameters_from_yaml(self, file_path):
        with open(file_path, 'r') as file:
            grid_parameters = yaml.safe_load(file)
        return grid_parameters

    def run_experiment(self):
        # Load Dataframe
        df = pd.read_csv("initialization/ground_20.csv")

        v_real = df['voltage'].values
        i_real = df['current'].values
        time = df['time']

        # Load YAML
        grid_parameters = self.load_grid_parameters_from_yaml('initialization/grid_parameters')
        electrical_params = self.load_electrical_params_from_yaml('initialization/electrical_params')
        thermal_params = self.load_thermal_params_from_yaml('initialization/thermal_params')
        models_config = [electrical_params, thermal_params]
        battery_options = self.load_battery_options_from_yaml('initialization/battery_options')

        load_var = 'current'

        battery = BatteryEnergyStorageSystem(
            models_config=models_config,
            battery_options=battery_options,
            input_var=load_var
        )

        #reset_info = {key: electrical_params['components'][key]['scalar'] for key in
        #            electrical_params['components'].keys()}
        reset_info = {'electricala_params': electrical_params, 'thermal_params': thermal_params}
        battery.reset(reset_info)

        battery.init({'dissipated_heat' : 0 })

        history_theta = list()
        nominal_clusters = dict()

        elapsed_time = 0
        dt = 1
        soc = battery.soc_series[-1]
        temp = battery._thermal_model.get_temp_series(-1)

        grid = Grid(grid_parameters, soc, temp)
        start = 0
        v_optimizer = list()
        temp_optimizer = list()

        battery_results = battery.get_last_results()
        optimizer = Optimizer(models_config=models_config, battery_options=battery_options, load_var=load_var,
                              init_info=battery_results)

        for k, load in enumerate(i_real):
            if k < self.training_window:
                elapsed_time += dt
                battery.t_series.append(elapsed_time)
                dt = df['time'].iloc[k] - df['time'].iloc[k - 1] if k > 0 else 1.0
                #print(" k:", k, "is changed:", grid.is_changed_cell(soc, temp) )
                if (k % self.batch_size == 0 and k != 0)  or grid.is_changed_cell(soc, temp):

                    theta = optimizer.step(i_real=i_real[start:k], v_real=v_real[start:k], optimizer_method= self.optimizer_method,
                                           alpha=self.alpha,dt=dt, number_of_restarts= self.number_of_restarts)

                    history_theta.append(theta)
                    start = k
                    v_optimizer = v_optimizer + optimizer.get_v_hat()
                    temp_optimizer = temp_optimizer + optimizer.get_t_hat()
                    #battery_results = battery.get_last_results(), do I need this ???

                    # TODO: REMOVE !
                    battery._electrical_model.r0.resistance = theta['r0']
                    battery._electrical_model.rc.resistance = theta['rc']
                    battery._electrical_model.rc.capacity = theta['c']

                    # cluster population:
                    if grid.current_cell not in nominal_clusters:
                        nominal_clusters[grid.current_cell] = Cluster()
                    nominal_clusters[grid.current_cell].add( np.array([theta['r0'],theta['rc'], theta['c']]) )

                battery.step(load, dt, k)
                soc = battery.soc_series[-1]
                temp = battery._thermal_model.get_temp_series(-1)

        # printing phase:
        results = battery.build_results_table()
        results = results['operations']
        # Create a figure and two subplots, one for voltage and one for temperature
        fig, (ax1, ax2) = plt.subplots(2, 1)

        # Plot voltage data
        #v_dt_updated = self.extract_v(status_series)
        ax1.plot(results['voltage'], label='v')
        ax1.plot(df['voltage'][0:len(results['voltage'])], label='ground')
        ax1.plot(v_optimizer[0:len(results['voltage'])], label='v_optimizer')
        ax1.legend()

        # Plot temperature data
        #temp_dt_updated = self.extract_temp(status_series)
        ax2.plot(results['temperature'], label='temperature')
        ax2.plot(df['temperature'][0:len(results['temperature'])], label='ground')
        ax2.plot(temp_optimizer[0:len(results['temperature'])], label='temp_optimizer')
        ax2.legend()

        plt.show()


        #Clusters info:
        data = []
        print("Nominal Clusters info")
        for cell, cluster in nominal_clusters.items():
            print("cell: ", cell)
            cluster.compute_centroid()
            print("centroid: ", cluster.centroid)
            cluster.compute_covariance_matrix()
            print("Covariance-Matrix: ", cluster.covariance_matrix)
            print("cluster itself: ", cluster.get_parameters())

            # Append the information to the data list:
            data.append({
                "cell": cell,
                "centroid": cluster.centroid,
                "covariance_matrix": cluster.covariance_matrix,
                "cluster": cluster.parameters
            })


        #save data into a csv:
        df_nominal_clusters = pd.DataFrame(data).set_index("cell")
        csv_file = "../../notebooks/online_learning/saved_results/nominal_clusters.csv"
        df_nominal_clusters.to_csv(csv_file)







