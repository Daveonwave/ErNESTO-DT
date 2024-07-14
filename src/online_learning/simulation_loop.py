from src.digital_twin.bess import BatteryEnergyStorageSystem
from src.online_learning.cluster import Cluster
from src.online_learning.grid import Grid
from src.online_learning.optimizer import Optimizer
from src.online_learning.utils import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class Simulation:
    def __init__(self, alpha, batch_size, optimizer_method, training_window,
                 save_results, number_of_restarts):
        self.alpha = alpha
        self.batch_size = batch_size
        self.optimizer_method = optimizer_method
        self.training_window = training_window
        self.save_results = save_results
        self.number_of_restarts = number_of_restarts
        self.cluster_info = False
        self.cluster_population = False

    def set_theta_parameters(self,battery ,theta):
        battery._electrical_model.r0.resistance = theta['r0']
        battery._electrical_model.rc.resistance = theta['rc']
        battery._electrical_model.rc.capacity = theta['c']

    def run_experiment(self):

        relative_path_ground = os.path.join('..', '..', 'src', 'online_learning', '../../data/initialization', 'ground_20.csv')
        path_ground = utils.get_absolute_path(relative_path_ground)

        # Load Dataframe
        df = pd.read_csv(path_ground)

        v_real = df['voltage'].values
        i_real = df['current'].values
        t_real = df['temperature'].values

        # Load YAML:
        relative_path_grid = os.path.join('..','..','src','online_learning', '../../data/initialization', 'grid_parameters')
        path_grid = utils.get_absolute_path(relative_path_grid)
        grid_parameters = utils.load_from_yaml(path_grid)

        relative_path_electrical_params = os.path.join('..','..','src','online_learning',
                                                       '../../data/initialization', 'electrical_params')
        path_electrical_params = utils.get_absolute_path(relative_path_electrical_params)
        electrical_params = utils.load_from_yaml(path_electrical_params)

        relative_path_thermal_params = os.path.join('..', '..', 'src', 'online_learning',
                                                    '../../data/initialization', 'thermal_params')
        path_thermal_params = utils.get_absolute_path(relative_path_thermal_params)

        thermal_params = utils.load_from_yaml(path_thermal_params)
        models_config = [electrical_params, thermal_params]

        relative_path_battery_options = os.path.join('..', '..', 'src', 'online_learning',
                                                     '../../data/initialization', 'battery_options')
        path_battery_options = utils.get_absolute_path(relative_path_battery_options)

        battery_options = utils.load_from_yaml(path_battery_options)

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

                if (k % self.batch_size == 0 and k != 0)  or grid.is_changed_cell(soc, temp):

                    theta = optimizer.step(i_real=i_real[start:k], v_real=v_real[start:k],
                                           t_real=t_real[start:k],optimizer_method= self.optimizer_method,
                                           alpha=self.alpha,dt=dt, number_of_restarts= self.number_of_restarts)


                    history_theta.append(theta)
                    start = k
                    v_optimizer = v_optimizer + optimizer.get_v_hat()
                    temp_optimizer = temp_optimizer + optimizer.get_t_hat()
                    #battery_results = battery.get_last_results()# do I need this ???

                    #self.set_theta_parameters(battery=battery, theta=theta)

                    # cluster population:
                    #if self.cluster_population:
                    #    if grid.current_cell not in nominal_clusters:
                    #        nominal_clusters[grid.current_cell] = Cluster()
                    #    nominal_clusters[grid.current_cell].add( np.array([theta['r0'],theta['rc'], theta['c']]) )

                battery.step(load, dt, k)
                soc = battery.soc_series[-1]
                temp = battery._thermal_model.get_temp_series(-1)

        #save
        if self.save_results:
            pass



