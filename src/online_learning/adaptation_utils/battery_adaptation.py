from src.digital_twin.bess import BatteryEnergyStorageSystem
from src.online_learning.cluster import Cluster
from src.online_learning.grid import Grid
from src.online_learning.optimizer import Optimizer
from src.online_learning.adaptation_utils.adaptation import fault_cluster_creation
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
        self.plot = False

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

    def load_cluster(self, csv_string):

        data = pd.read_csv(csv_string)

        array_elements = data[['R_0', 'R_1', 'C_1']].to_numpy()

        list_of_arrays = [np.array(row) for row in array_elements]

        return list_of_arrays


    # REMEMBER THAT THIS ALG. ASSUMES ALL THE NOMINAL CLUSTERS ARE POPULATED
    def run_experiment(self):
        
        work_dir = "src/online_learning/"

        # Load Dataframe
        df = pd.read_csv(work_dir + "initialization/ground_20.csv")

        v_real = df['voltage'].values
        i_real = df['current'].values
        t_real = df['temperature'].values
        time = df['time']

        # Load YAML
        grid_parameters = self.load_grid_parameters_from_yaml(work_dir + 'initialization/grid_parameters')
        electrical_params = self.load_electrical_params_from_yaml(work_dir + 'initialization/electrical_params')
        thermal_params = self.load_thermal_params_from_yaml(work_dir + 'initialization/thermal_params')
        models_config = [electrical_params, thermal_params]
        battery_options = self.load_battery_options_from_yaml(work_dir + 'initialization/battery_options')

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
        nominal_clusters = dict()
        outliers_sets = dict()

        file_names = ["phi_one.csv", "phi_two.csv", "phi_three.csv", "phi_four.csv"]

        for i, file_name in enumerate(file_names):
            phi = self.load_cluster(file_name)
            nominal_clusters[i] = Cluster()
            nominal_clusters[i].set(phi)
            nominal_clusters[i].compute_covariance_matrix()
            nominal_clusters[i].compute_centroid()
            print(nominal_clusters[i].get_parameters())
            print(nominal_clusters[i].get_centroid())
            print(len(phi))

        elapsed_time = 0
        dt = 1

        soc = battery.soc_series[-1]
        temp = battery._thermal_model.get_temp_series(-1)

        grid = Grid(grid_parameters, soc, temp)
        start = 0
        v_optimizer = list()
        temp_optimizer = list()
        phi_hat = None

        battery_results = battery.get_last_results()
        optimizer = Optimizer(models_config=models_config, battery_options=battery_options, load_var=load_var,
                              init_info=battery_results)

        for k, load in enumerate(i_real):
                elapsed_time += dt
                battery.t_series.append(elapsed_time)
                dt = df['time'].iloc[k] - df['time'].iloc[k - 1] if k > 0 else 1.0
                # print(" k:", k, "is changed:", grid.is_changed_cell(soc, temp) )

                if (k % self.batch_size == 0 and k != 0) or grid.is_changed_cell(soc, temp):

                    theta = optimizer.step(i_real=i_real[start:k], v_real=v_real[start:k], t_real= t_real[start:k]
                                           ,optimizer_method=self.optimizer_method,
                                           alpha=self.alpha, dt=dt, number_of_restarts=self.number_of_restarts)

                    history_theta.append(theta)
                    start = k
                    v_optimizer = v_optimizer + optimizer.get_v_hat()
                    temp_optimizer = temp_optimizer + optimizer.get_t_hat()


                    # TODO: IS IT MEANINGFUL?
                    # TODO: THEN REMOVE!
                    if grid.current_cell in nominal_clusters:
                       #print("------------------------------debug")
                       #print(np.shape(theta))
                       #print(theta)
                       theta_array = np.array([theta['r0'],theta['rc'],theta['c']])
                       if nominal_clusters[grid.current_cell].contains_within(theta_array):
                          #print("------------------------------------------------------------------")
                          print("the prediction is contained in a cluster, hence update the cluster")
                          #print("------------------------------------------------------------------")
                          nominal_clusters[grid.current_cell].add(theta_array)
                          nominal_clusters[grid.current_cell].compute_centroid()
                          nominal_clusters[grid.current_cell].compute_covariance_matrix()

                       else:
                          if grid.current_cell not in outliers_sets:
                              outliers_sets[grid.current_cell] = list()
                          outliers_sets[grid.current_cell].append(theta)

                       #print("nominal_clusters[grid.current_cell]:",
                       #      nominal_clusters[grid.current_cell].get_parameters())
                       #print(type(nominal_clusters[grid.current_cell].get_parameters()))

                       #TODO: CHECK MEANING ?
                       if grid.current_cell in outliers_sets:
                         phi_hat = fault_cluster_creation( cluster_parameters=nominal_clusters[grid.current_cell].get_parameters(),
                                                         outliers_set= outliers_sets[grid.current_cell] )

                       if phi_hat is not None:
                          print("fault cluster creation works !!!")
                          nominal_clusters[grid.current_cell] = phi_hat

                       centroid = nominal_clusters[grid.current_cell].centroid
                       #print(centroid)
                    # TODO: UNDERSTAND WHY THIS GIVES YOU PROBLEM WITH THE STEP CURENT DRIVEN ???
                    battery._electrical_model.r0.resistance = centroid[0]
                    battery._electrical_model.rc.resistance = centroid[1]
                    battery._electrical_model.rc.capacity = centroid[2]

                battery.step(load, dt, k)
                soc = battery.soc_series[-1]
                temp = battery._thermal_model.get_temp_series(-1)



        results = battery.build_results_table()
        results = results['operations']
        if self.plot:
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
        """
        if self.save_results:
            #saving results:
            #dict_volt = {"v":results['voltage'],
            #             "ground": df['voltage'][0:len(results['voltage'])],
            #             "v_optimizer": v_optimizer[0:len(results['voltage'])]}
            #df_volt = pd.DataFrame(dict_volt)
            #csv_file = 'run1/voltages.csv'
            #df_volt.to_csv(csv_file, index=False)

            #dict_temp = {"temperature": results['temperature'],
            #             "ground": df['temperature'][0:len(results['temperature'])],
            #             "temp_optimizer": temp_optimizer[0:len(results['temperature'])]
            #             }
            #df_temp = pd.DataFrame(dict_temp)
            #csv_file = 'run1/temperatures.csv'
            #df_temp.to_csv(csv_file, index=False)

            outliers_dir = 'run1/outliers'
            os.makedirs(outliers_dir, exist_ok=True)

            # Iterate over each key-value pair in the outliers_sets dictionary
            for key, value in outliers_sets.items():
                # Convert set to list for DataFrame compatibility (optional)
                value_list = list(value)

                # Create a DataFrame for the current object (value)
                df = pd.DataFrame({key: value_list})

                # Generate the CSV file path based on the key, within the outliers directory
                csv_file = os.path.join(outliers_dir, f'{key}.csv')

                # Save the DataFrame to CSV without index
                df.to_csv(csv_file, index=False)

            # Iterate over each key-value pair in the nominal_clusters dictionary
            for key, value in nominal_clusters.items():
                # Ensure the value is an iterable (e.g., list)
                value_list = list(value.get_parameters())

                # Create a DataFrame for the current object (value)
                df = pd.DataFrame({key: value_list})

                # Generate the CSV file path based on the key
                csv_file = os.path.join('run1', f'{key}.csv')

                # Save the DataFrame to CSV without index
                df.to_csv(csv_file, index=False)
        """




