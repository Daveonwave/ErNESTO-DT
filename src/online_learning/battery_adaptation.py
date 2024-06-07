from src.digital_twin.bess import BatteryEnergyStorageSystem
from src.online_learning.cluster import Cluster
from src.online_learning.grid import Grid
from src.online_learning.optimizer import Optimizer
from src.online_learning.adaptation import fault_cluster_creation
from notebooks.online_learning import saved_results
import pandas as pd
import numpy as np
import yaml
import ast
import matplotlib.pyplot as plt


class Battery_Adaptation:
    def __init__(self, alpha, batch_size, optimizer_method, save_results, number_of_restarts):
        self.alpha = alpha
        self.batch_size = batch_size
        self.optimizer_method = optimizer_method
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

    def convert_to_array_list(self, cluster_string):
        # Convert the string representation to an actual list using ast.literal_eval
        # Then convert each element back to a numpy array
        try:
            # Evaluate the string to a list
            cluster_list = ast.literal_eval(self, cluster_string)
            # Ensure each item in the list is a numpy array
            return [np.array(item) for item in cluster_list]
        except (ValueError, SyntaxError):
            # Handle potential errors in string conversion
            return []

    def get_nominal_clusters(self):
        df = pd.read_csv("initialization/nominal_clusters.csv")
        dict = {}
        for i in df.keys():
            #cluster = Cluster()
            #dict[i] =
            print(i)
            print(df.keys())


        exit()
        return None


    # REMEMBER THAT THIS ALG. ASSUMES ALL THE NOMINAL CLUSTERS ARE POPULATED
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

        # reset_info = {key: electrical_params['components'][key]['scalar'] for key in
        #            electrical_params['components'].keys()}
        reset_info = {'electricala_params': electrical_params, 'thermal_params': thermal_params}
        battery.reset(reset_info)
        battery.init({'dissipated_heat': 0})

        history_theta = list()
        nominal_clusters = self.get_nominal_clusters()
        outliers_sets = dict()

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
                elapsed_time += dt
                battery.t_series.append(elapsed_time)
                dt = df['time'].iloc[k] - df['time'].iloc[k - 1] if k > 0 else 1.0
                # print(" k:", k, "is changed:", grid.is_changed_cell(soc, temp) )

                if (k % self.batch_size == 0 and k != 0) or grid.is_changed_cell(soc, temp):

                    theta = optimizer.step(i_real=i_real[start:k], v_real=v_real[start:k],
                                           optimizer_method=self.optimizer_method,
                                           alpha=self.alpha, dt=dt, number_of_restarts=self.number_of_restarts)

                    history_theta.append(theta)
                    start = k
                    v_optimizer = v_optimizer + optimizer.get_v_hat()
                    temp_optimizer = temp_optimizer + optimizer.get_t_hat()


                    # TODO: IS IT MEANINGFUL?
                    # TODO: THEN REMOVE!
                    if grid.current_cell in nominal_clusters:
                       if nominal_clusters[grid.current_cell].contains_within(theta):
                          print("------------------------------------------------------------------")
                          print("the prediction is contained in a cluster, hence update the cluster")
                          print("------------------------------------------------------------------")
                          nominal_clusters[grid.current_cell].add(theta)
                          nominal_clusters[grid.current_cell].compute_centroid()
                          nominal_clusters[grid.current_cell].compute_covariance_matrix()

                       if grid.current_cell not in outliers_sets:
                          outliers_sets[grid.current_cell] = list()
                       outliers_sets[grid.current_cell].append(theta)

                       print("nominal_clusters[grid.current_cell]:",
                             nominal_clusters[grid.current_cell].get_parameters())
                       print(type(nominal_clusters[grid.current_cell].get_parameters()))
                       exit()

                       phi_hat = fault_cluster_creation( cluster_parameters=nominal_clusters[grid.current_cell].get_parameters(),
                                                         outliers_set= outliers_sets[grid.current_cell] )

                       if phi_hat is not None:
                          print("fault cluster creation works !!!")
                          nominal_clusters[grid.current_cell] = phi_hat

                       centroid = nominal_clusters[grid.current_cell].centroid
                       print(centroid)
                    # TODO: UNDERSTAND WHY THIS GIVES YOU PROBLEM WITH THE STEP CURENT DRIVEN ???
                    #battery._electrical_model.r0.resistance = centroid[0]
                    #battery._electrical_model.rc.resistance = centroid[1]
                    #battery._electrical_model.rc.capacity = centroid[2]

                battery.step(load, dt, k)
                soc = battery.soc_series[-1]
                temp = battery._thermal_model.get_temp_series(-1)





