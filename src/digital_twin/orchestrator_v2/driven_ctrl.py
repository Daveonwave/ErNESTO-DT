import pandas as pd
import logging
from tqdm import tqdm

from src.digital_twin.orchestrator.base_manager import GeneralPurposeManager
from src.digital_twin.bess import BatteryEnergyStorageSystem
from src.preprocessing.data_preparation import load_data_from_csv, validate_parameters_unit, sync_data_with_step
from src.preprocessing.schema import read_yaml
from src.postprocessing.metrics import compute_metrics

logger = logging.getLogger('DT_ernesto')


class DrivenController:
    """
    Handler of the Compared Simulation experiment.
    -----------------------------------------
    The simulator is conceived to be the orchestrator and the brain of the specified experiment.

    From here, all the kinds of data (input, output, config) are delivered to their consumer hubs, the
    environment is instantiated and the instructions related to the simulation mode chosen by the user are provided.
    """
    def __init__(self, data_loader):
        self._mode = "simulation"
        logger.info("Instantiated {} class as experiment orchestrator".format(self.__class__.__name__))

        # Prepare ground preprocessing for input and validation
        self._input_var = self._settings['ground_data']['load']
        self._ground_vars = [item['var'] for item in self._settings['ground_data']['vars']]
        self._stepsize = self._settings['timestep'] if 'timestep' in self._settings else None

        if self._ground_folder is None or 'ground_data' not in self._settings:
            raise KeyError("Ground data information not provided: maybe you need to run a What-If experiment")

        self._ground_times, self._ground_data = (
            load_data_from_csv(csv_file=self._ground_folder / self._settings['ground_data']['file'],
                               vars_to_retrieve=self._settings['ground_data']['vars'],
                               time_format=self._settings['ground_data']['time_format'],
                               iterations=self._settings['iterations'])
        )

        if self._stepsize is not None:
            self._ground_times, self._ground_data = sync_data_with_step(times=self._ground_times.copy(),
                                                                        data=self._ground_data.copy(),
                                                                        sim_step=self._stepsize)

        # Time options
        self._duration = self._ground_times[-1] - self._ground_times[0]
        self._elapsed_time = 0
        self._done = False

        # Validate battery parameters unit
        self._settings['battery']['params'] = validate_parameters_unit(self._settings['battery']['params'])

        # Instantiate the BESS environment
        self._battery = BatteryEnergyStorageSystem(
            models_config=self._models_configs,
            battery_options=self._settings['battery'],
            input_var=self._input_var,
            check_soh_every=self._settings['check_soh_every'] if 'check_soh_every' in self._settings else None,
            ground_data=self._ground_data
        )

    def run(self):
        """
        Simulation experiment:
        """
        logger.info("'Compared Simulation' started...")
        self._battery.reset()
        self._battery.init()

        if self._stepsize is not None:
            self._imposed_run()
        else:
            self._fitted_run()

        self.done = True
        self._results = self._battery.build_results_table()

        # Cleaned ground data to output as csv
        self._ground_data['time'] = [t - self._ground_times[0] for t in self._ground_times]
        self._results['ground'] = self._ground_data

        self._output_results(results=self._results, summary=self._get_summary())
        if self._make_plots:
            self._prepare_plots()

        # Rainflow:   0.0005420391580281958   6.17753705259805
        # Streamflow: 0.0005420576069465888   6.177507467564798
        # Streamflow (reset 3000):  0.00010375468187362458    1.9731715967437136

    def _imposed_run(self):
        """
        Run experiment imposing a given timestep. This is done by a manipulation of ground data in order to match the
        size that the dataset should have with the given timestep, through data augmentation or reduction.
        """
        k = 0
        dt = self._stepsize
        pbar = tqdm(total=int(self._duration), position=0, leave=True)

        # Main loop of the simulation
        while self._elapsed_time <= self._duration and not self._done:

            # Having manipulated the ground dataset before, we can be sure about the sync with elapsed time
            self._battery.step(load=self._ground_data[self._input_var][k], dt=dt, k=k)
            self._battery.t_series.append(self._elapsed_time)
            self._elapsed_time += dt

            k += 1
            pbar.update(dt)

            if k < len(self._ground_times):
                dt = round(self._ground_times[k] - self._ground_times[k - 1], 2)
            else:
                self._done = True

        pbar.close()
        logger.info("'Compared Simulation' ended without errors!")

        # Cleaning all remaining data (e.g. with dt=0 at the end of the simulation)
        self._ground_times = self._ground_times[:k]
        for key in self._ground_data.keys():
            self._ground_data[key] = self._ground_data[key][:k]

    def _fitted_run(self):
        """
        Run experiment fitting the timestep on the sampling time of ground data.
        If the ground timestep is 0 the current step is skipped. If the ground timestep is too high (still need to
        define what it means), we have a hole in the ground dataset, translated for convenience in a period where the
        battery has been turned off.
        """
        k = 0
        dt = 1
        pbar = tqdm(total=int(self._duration), position=0, leave=True)

        # Main loop of the simulation
        while self._elapsed_time <= self._duration and not self._done:

            # No progress in the simulation due to preprocessing timestamp
            if dt == 0:
                self._ground_times.pop(k)
                [self._ground_data[key].pop(k) for key in self._ground_data.keys()]
            else:
                # When the dt overcomes the step size it means that there is a lack of data in load profile, thus we
                # consider the battery as turned off.
                # todo: cambiare con valore adeguato
                if dt > 1:
                    self._battery.load_var = 'current'
                    self._battery.step(load=0, dt=dt, k=k)
                    self._battery.load_var = self._input_var
                # Normal operating condition of the battery
                else:
                    self._battery.step(load=self._ground_data[self._input_var][k], dt=dt, k=k)

                self._battery.t_series.append(self._elapsed_time)
                self._elapsed_time += dt
                k += 1

            pbar.update(dt)
            if k < len(self._ground_times):
                dt = round(self._ground_times[k] - self._ground_times[k - 1], 2)
            else:
                self._done = True

        pbar.close()
        logger.info("'Compared Simulation' ended without errors!")

        # Cleaning all remaining data (e.g. with dt=0 at the end of the simulation)
        self._ground_times = self._ground_times[:k]
        for key in self._ground_data.keys():
            self._ground_data[key] = self._ground_data[key][:k]

    def evaluate(self):
        res = compute_metrics(ground=self._results['ground'],
                              simulated=self._results['operations'],
                              vars=['voltage', 'temperature'],
                              metrics=['mse', 'mae', 'mape', 'max_abs_err'],
                              steps=None)
        self._output_metrics(res)

    def _get_summary(self):
        """
        Get simulation summary with important information
        TODO: update when will be added new features
        """
        return {'experiment': self._settings['experiment_name'],
                'description': self._settings['description'],
                'goal': self._settings['goal'],
                'load': self._settings['ground_data']['load'],
                'time': self._elapsed_time,
                'timestep': self._settings['timestep'],
                'iterations': self._settings['iterations'],
                'battery': self._settings['battery']['params'],
                'initial_conditions': self._settings['battery']['init'],
                'models': [model.__class__.__name__ for model in self._battery.models]
                }

    def _prepare_plots(self):
        """
        # TODO: clean up this method -> add save plot img option (leggere da file var_to_plot)

        """
        # PLOT CONFRONTO O NON CONFRONTO A SECONDA CHE CI SIA GROUND O MENO
        var_to_plot = ['voltage', 'temperature', 'power', 'current']
        var_to_plot = self._ground_vars
        #var_to_plot = ['voltage', 'temperature', 'current']

        df_simulated = pd.DataFrame(data=self._results['operations'], columns=self._results['operations'].keys())

        # Gather ground data in a unique dataframe
        df_ground = pd.DataFrame(self._ground_data)

        df_ground['timestamp'] = self._ground_times
        df_ground['timestamp'] = df_ground['timestamp'] - df_ground['timestamp'][0]

        # Save information for each different kind of plot
        for var in var_to_plot:
            plot_dict = {
                'type': "compared",
                'dfs': [df_simulated.iloc[1:], df_ground],
                'variables': [var, var],
                'labels': ['Simulated', 'Ground'],
                'x_axes': ['time', 'timestamp'],
                'title': var
            }
            self._plot_info.append(plot_dict)
            del plot_dict

        self._save_plots()
