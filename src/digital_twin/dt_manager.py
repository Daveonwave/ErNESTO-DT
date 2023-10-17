import yaml
import os
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from rich.pretty import pretty_repr

from src.digital_twin.bess import BatteryEnergyStorageSystem
from src.visualization.fast_plots import plot_compared_data
from src.data.processing import load_data_from_csv, validate_parameters_unit

logger = logging.getLogger('DT_logger')


class GeneralPurposeManager:
    """
    Handler of the Digital Twin experiment.
    -----------------------------------------
    The simulator is conceived to be the orchestrator and the brain of the specified simulation/experiment.

    From here, all the kinds of data (input, output, configuration) are delivered to their consumer hubs, the
    environment is instantiated and the instructions related to the simulation mode chosen by the user are provided.
    """
    @classmethod
    def get_instance(cls, mode):
        logger.info("Starting '{}' experiment...".format(mode))
        return next(c for c in cls.__subclasses__() if mode in c.__name__.lower())

    def __init__(self,
                 experiment_config,
                 time_options,
                 save_results=False,
                 plot_results=False,
                 **data_folders
                 ):
        """

        """
        # Store paths for all different kind of data
        self.config_data_path = data_folders['config_data']
        self.output_data_path = data_folders['output_data']
        self.ground_data_path = data_folders['ground_data']

        with open(self.config_data_path / Path(experiment_config), 'r') as fin:
            self.experiment_config = yaml.safe_load(fin)

        # Custom number of iterations or given by data
        self._iterations = None
        if time_options['iterations']:
            self._iterations = time_options['iterations']
        self._timestep = time_options['timestep']

        # Output results and visualization
        self.save_results = save_results
        self.plot_results = plot_results
        self.output_folder = (self.output_data_path / self.experiment_config['experiment_folder'] /
                              str(datetime.now().strftime('%Y-%m-%d_%H-%M')))

    def run(self):
        pass

    def render(self):
        pass

    def save_results(self):
        pass

    def _save_plots(self):
        pass


class SimulationManager(GeneralPurposeManager):
    """

    """
    def __init__(self,
                 models,
                 experiment_config,
                 time_options,
                 save_results=False,
                 plot_results=False,
                 **data_folders
                 ):
        super().__init__(experiment_config, time_options, save_results, plot_results, **data_folders)
        logger.info("Instantiated {} class as experiment orchestrator".format(self.__class__.__name__))

        # Prepare ground data for input and validation
        self.input_var = self.experiment_config['ground_data']['load']
        self.ground_data, self.ground_times = load_data_from_csv(
            csv_file=self.ground_data_path / self.experiment_config['ground_data']['file'],
            vars_to_retrieve=self.experiment_config['ground_data']['vars'],
            iterations=self._iterations)

        # Time options
        self._duration = self.ground_times[-1] - self.ground_times[0]
        # TODO: understand how to handle sampling time if different from csv data sampling interval => interpolation
        self._elapsed_time = 0
        self.done = False

        models_config_files = []
        for model in models:
            model_file = Path(self.experiment_config['models'][model]['category'] + '/' +
                              self.experiment_config['models'][model]['file'])
            models_config_files.append(self.config_data_path / model_file)

        self._initial_conditions = self.experiment_config['initial_conditions']

        # Instantiate the BESS environment
        self._battery = BatteryEnergyStorageSystem(
            models_config_files=models_config_files,
            battery_options=validate_parameters_unit(self.experiment_config['battery']),
            input_var=self.input_var,
            sign_convention=self.experiment_config['sign_convention']
        )

    def run(self):
        self._battery.reset_data()
        self._battery.simulation_init(initial_conditions=self._initial_conditions)

        load = self.ground_data[self.input_var].copy()
        times = self.ground_times.copy()

        k = 0
        dt = 1
        pbar = tqdm(total=int(self._duration), position=0, leave=True)

        # Main loop of the simulation
        while self._elapsed_time < self._duration:

            # No progress in the simulation due to data timestamp
            if dt == 0:
                load.pop(k)
                times.pop(k)
            else:
                self._battery.simulation_step(load=load[k], dt=dt, k=k - 1)
                self._battery.t_series.append(self._elapsed_time)
                self._elapsed_time += dt
                k += 1

            pbar.update(dt)
            dt = times[k] - times[k - 1]

        pbar.close()
        self.done = True

        self._output_results()
        self._show_results()

    def render(self):
        # TODO: implement this
        raise NotImplementedError()

    def output_results(self):
        # TODO: implement this
        raise NotImplementedError()

    def _output_results(self):
        """

        """
        if self.save_results:
            try:
                os.makedirs(self.output_folder, exist_ok=True)
            except:
                raise NotADirectoryError("It's not possible to create directory {}!".format(self.output_folder))

            self._store_results()
            self._experiment_summary(store=True)
        else:
            self._experiment_summary(store=False)

    def _store_results(self):
        """
        # TODO: save results
        """
        self._battery.save_data(output_file=self.output_folder, file_name='dataset.csv')

    def _experiment_summary(self, store=False):
        """
        Output a summary of the current experiment to keep track of the configuration settings
        TODO: update when will be added new features
        """
        summary = {'experiment': self.experiment_config['experiment_name'],
                   'description': self.experiment_config['description'],
                   'goal': self.experiment_config['goal'],
                   'load': self.experiment_config['ground_data']['load'],
                   'time': self._iterations * self._timestep,
                   'battery': self.experiment_config['battery'],
                   'initial_conditions': self.experiment_config['initial_conditions'],
                   'models': [model.__class__.__name__ for model in self._battery.models],
                   }

        if store:
            summary_file = self.output_folder / "summary.txt"
            with open(summary_file, 'w') as f:
                for key, value in summary.items():
                    f.write('%s: %s\n' % (key, value))

        else:
            logger.info(pretty_repr(summary))

    def _show_results(self):
        """

        """
        if self.plot_results:
            var_to_plot = ['Voltage [V]', 'Temperature [degC]', 'Power [W]']
            self._fast_plot(var_to_plot=var_to_plot)

    def _fast_plot(self, var_to_plot: list):
        """
        # TODO: clean up this method -> add save plot img option
        """
        df = self._battery.build_results_table()
        # print(df['Power [W]'])

        df_ground = pd.read_csv(self.ground_data_path / self.experiment_config['ground_data']['file'], encoding='unicode_escape')
        if self._iterations:
            df_ground = df_ground.iloc[:self._iterations]

        df_ground['Timestamp'] = pd.to_datetime(df_ground['Time'], format="%Y/%m/%d %H:%M:%S").values.astype(
            float) // 10 ** 9
        df_ground['Timestamp'] = df_ground['Timestamp'] - df_ground['Timestamp'][0]

        for var in var_to_plot:
            dfs = [df.iloc[1:], df_ground]
            var = [var, var]
            labels = ['Simulated', 'Ground']
            x_axes = ['Time', 'Timestamp']
            title = var

            plot_compared_data(dfs=dfs, variables=var, x_axes=x_axes, labels=labels, title=title)


class WhatIfManager(GeneralPurposeManager):
    """

    """
    def __init__(self,
                 models,
                 experiment_config,
                 save_results=False,
                 plot_results=False,
                 **data_folders
                 ):
        super().__init__(experiment_config, save_results, plot_results, **data_folders)


class LearningManager(GeneralPurposeManager):
    """

    """
    def __init__(self,
                 models,
                 experiment_config,
                 save_results=False,
                 plot_results=False,
                 **data_folders
                 ):
        super().__init__(experiment_config, save_results, plot_results, **data_folders)


class OptimizationManager(GeneralPurposeManager):
    """

    """
    def __init__(self,
                 models,
                 experiment_config,
                 save_results=False,
                 plot_results=False,
                 **data_folders
                 ):
        super().__init__(experiment_config, save_results, plot_results, **data_folders)