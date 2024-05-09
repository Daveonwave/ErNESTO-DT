from src.digital_twin.bess import BatteryEnergyStorageSystem
from notebooks.online_learning.cluster import Cluster
from notebooks.online_learning.grid import Grid
from notebooks.online_learning.optimizer import Optimizer
import pandas as pd
import yaml
import numpy as np
import yaml
import matplotlib.pyplot as plt

class Simulation:
    def __init__(self, alpha, batch_size, optimizer, training_window):
        self.alpha = alpha
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.training_window = training_window

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

    def update_settings(self, battery, battery_options, k):
        battery_options['init']['soc'] = battery.soc_series[-1]
        battery_options['init']['temperature'] = battery._thermal_model.get_temp_series(k)
        battery_options['init']['current'] = battery._electrical_model.get_v_series(k)
        battery_options['init']['voltage'] = battery._electrical_model.get_i_series(k)

    def update_electrical_params(self, theta, models_config):
        # 0 stands for electrical_params
        models_config[0]['r0'] = theta[0]
        models_config[0]['r1'] = theta[1]
        models_config[0]['c'] = theta[2]

    def extract_v(self, status_series):
        v = []
        for item in status_series:
            # print(item)
            v.extend(item['operations']['voltage'])
        return v

    def extract_temp(self, status_series):
        temp = []
        for item in status_series:
            temp.extend(item['operations']['temperature'])
        return temp

    def run_experiment(self):
        # Load Dataframe
        df = pd.read_csv("ground_20.csv")

        v_real = df['voltage'].values
        i_real = df['current'].values

        # Load YAML
        grid_parameters = self.load_grid_parameters_from_yaml('grid_parameters')
        electrical_params = self.load_electrical_params_from_yaml('electrical_params')
        thermal_params = self.load_thermal_params_from_yaml('thermal_params')
        models_config = [electrical_params, thermal_params]
        battery_options = self.load_battery_options_from_yaml('battery_options')

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
        #heat = battery._electrical_model.compute_generated_heat()
        # TODO: E' UNA PEZZA PER VEDERE SE VA !
        battery.init({'dissipated_heat' : 0 })

        nominal_clusters = dict()
        optimizer = Optimizer( models_config=models_config,battery_options=battery_options,load_var= load_var)
        elapsed_time = 0
        dt = 1
        soc = battery.soc_series[-1]
        temp = battery._thermal_model.get_temp_series(-1)
        #print(temp)
        grid = Grid(grid_parameters, soc, temp)
        start = 0
        status_series = list()

        for k, load in enumerate(i_real):
            if k < self.training_window:
                elapsed_time += dt
                dt = df['time'].iloc[k] - df['time'].iloc[k - 1] if k > 0 else 1.0
                if k % self.batch_size == 0 or grid.is_changed_cell(soc, temp):
                    battery_results = battery.get_last_results()
                    theta = optimizer.step(i_real=i_real[start:k], v_real=v_real[start:k], init_info= battery_results, dt=dt)
                    #self.update_electrical_params(theta, settings['models_config'])
                    #theta update
                    battery._electrical_model.r0.resistance = theta['r0']
                    battery._electrical_model.rc.resistance = theta['rc']
                    battery._electrical_model.rc.capacity = theta['c']

                    start = k
                    status_series.append(optimizer.get_status())

                    # cluster population
                    if grid.current_cell not in nominal_clusters:
                        nominal_clusters[grid.current_cell] = Cluster()
                    nominal_clusters[grid.current_cell].add( np.array([theta['r0'],theta['rc'], theta['c']]) )
                    # end cluster population

                battery.step(load, dt, k)
                #self.update_settings(battery, settings['battery_options'], k)

                soc = battery.soc_series[-1]
                temp = battery._thermal_model.get_temp_series(-1)
            # end of training phase:
        # TODO: IMPLEMENT THE ONLINE LEARNING ALGORITHM:

        # printing phase:
        results = battery.build_results_table()
        results = results['operations']
        # Create a figure and two subplots, one for voltage and one for temperature
        fig, (ax1, ax2) = plt.subplots(2, 1)

        # Plot voltage data
        v_dt_updated = self.extract_v(status_series)
        ax1.plot(results['voltage'], label='v')
        ax1.plot(df['voltage'], label='ground')
        ax1.plot(v_dt_updated[100:len(results['voltage'])], label='v_dt_updated')
        ax1.legend()

        # Plot temperature data
        temp_dt_updated = self.extract_temp(status_series)
        ax2.plot(results['temperature'], label='temperature')
        ax2.plot(df['temperature'], label='ground')
        ax2.plot(temp_dt_updated[100:len(results['temperature'])], label='temp_dt_updated')
        ax2.legend()

        # Show the plots
        plt.show()

        #print Clusters info
        print("Nominal Clusters info")
        for cell, cluster in nominal_clusters.items():
            print("cell: ", cell)
            cluster.compute_centroid()
            print("centroid: ", cluster.centroid)
            cluster.compute_variance()
            print("variance :", cluster.variance)

        #save data into a csv
        df_voltage = pd.DataFrame({'voltage':results['voltage'], 'updated_voltage':v_dt_updated[0:len(results['voltage'])]})
        csv_file_voltage = "voltage_final.csv"
        df_voltage.to_csv(csv_file_voltage, index=False)

        df_temp = pd.DataFrame({'temperature': results['temperature'], 'updated_temp': temp_dt_updated[0:len(results['temperature'])]})
        csv_file_temp = "temp_final.csv"
        df_temp.to_csv(csv_file_temp, index=False)





