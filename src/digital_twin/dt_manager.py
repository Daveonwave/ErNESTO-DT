import yaml
import os
import pandas as pd
from datetime import datetime
from pathlib import Path
from src.digital_twin.bess import BatteryEnergyStorageSystem
from src.visualization.fast_plots import plot_compared_data


class DTManager:
    """
    Handler of the Digital Twin experiment.
    -----------------------------------------
    The simulator is conceived to be the orchestrator and the brain of the specified simulation/experiment.

    From here, all the kinds of data (input, output, configuration) are delivered to their consumer hubs, the
    environment is instantiated and the instructions related to the simulation mode chosen by the user are provided.
    """

    def __init__(self,
                 experiment_mode,
                 experiment_config,
                 models,
                 save_results=False,
                 plot_results=False,
                 **data_folders
                 ):
        """

        """
        # Store paths for all different kind of data
        self.config_data_path = data_folders['config_data']
        self.load_data_path = data_folders['load_data']
        self.output_data_path = data_folders['output_data']
        self.ground_data_path = data_folders['ground_data']

        # Configure the experiment for the required mode among {simulation, learning, whatif, optimization}
        self.experiment_mode = experiment_mode

        with open(self.config_data_path / Path(experiment_config), 'r') as fin:
            self.experiment_config = yaml.safe_load(fin)

        # TODO: check multiple electrical of the same type => PUT WARNING OR EXCEPTION
        models_config_files = []
        for model in models:
            model_file = Path(self.experiment_config['models'][model]['category'] + '/' +
                              self.experiment_config['models'][model]['file'])
            models_config_files.append(self.config_data_path / model_file)

        # Instantiate the BESS environment
        self.battery = BatteryEnergyStorageSystem(
            models_config_files=models_config_files,
            load_file=self.load_data_path / self.experiment_config['load_csv'],
            load_options=self.experiment_config['load'],
            time_options=self.experiment_config['time'],
            battery_options=self.experiment_config['battery'],
            initial_conditions=self.experiment_config['initial_conditions'],
            ground_file=self.ground_data_path / self.experiment_config['ground_csv'],
            ground_options=self.experiment_config['ground'],
            sign_convention=self.experiment_config['sign_convention'],
            units_checker=self.experiment_config['use_data_units']
        )

        self.save_results = save_results
        self.plot_results = plot_results
        self.output_folder = self.output_data_path / self.experiment_config['experiment_folder'] / \
                             str(datetime.now().strftime('%Y-%m-%d_%H-%M'))

    def run(self):
        """

        """
        if self.experiment_mode == 'simulation':
            self.battery.run_simulation()

            # for v_, ground in zip(v, self.battery.ground_data):
            # print(v_, ground)

            # print(self.battery.soc_series)

        self._output_results()
        self._show_results()

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
        self.battery.save_data(output_file=self.output_folder, file_name='dataset.csv')

    def _experiment_summary(self, store=False):
        """
        Output a summary of the current experiment to keep track of the configuration settings
        TODO: update when will be added new features
        """
        summary = {'experiment': self.experiment_config['experiment_name'],
                   'description': self.experiment_config['description'], 'goal': self.experiment_config['goal'],
                   'load': self.experiment_config['load']['label'], 'output': self.experiment_config['ground']['label'],
                   'time': self.experiment_config['time'],
                   'battery': self.experiment_config['battery'],
                   'initial_conditions': self.experiment_config['initial_conditions'],
                   'models': [model.__class__.__name__ for model in self.battery.models],
                   }

        if store:
            summary_file = self.output_folder / "summary.txt"
            with open(summary_file, 'w') as f:
                for key, value in summary.items():
                    f.write('%s: %s\n' % (key, value))

        else:
            for key, value in summary.items():
                print('%s: %s' % (key, value))

    def _show_results(self):
        """

        """
        if self.plot_results:
            var_to_plot = ['Voltage [V]', 'Temperature [C]', 'Power [W]']
            self._fast_plot(var_to_plot=var_to_plot)

    def _fast_plot(self, var_to_plot: list):
        """
        # TODO: clean up this method -> add save plot img option
        """
        df = self.battery.build_results_table()
        print(df['Power [W]'])

        df_ground = pd.read_csv(self.ground_data_path / self.experiment_config['ground_csv'], encoding='unicode_escape')
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
