import operator
import os
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from rich.pretty import pretty_repr

from src.digital_twin.handlers.base_manager import GeneralPurposeManager
from src.digital_twin.bess import BatteryEnergyStorageSystem
from src.visualization.plotter import plot_compared_data
from src.preprocessing.data_preparation import load_data_from_csv, validate_parameters_unit
from src.preprocessing.schema import read_yaml
from src.preprocessing.schedule.schedule import Schedule

logger = logging.getLogger('DT_logger')


class WhatIfManager(GeneralPurposeManager):
    """
    Handler of the What-if experiment.
    -----------------------------------------
    The simulator is conceived to be the orchestrator and the brain of the specified experiment.

    From here, all the kinds of data (input, output, config) are delivered to their consumer hubs, the
    environment is instantiated and the instructions related to the simulation mode chosen by the user are provided.
    """
    def __init__(self, **kwargs):
        self._mode = "whatif"
        logger.info("Instantiated {} class as experiment orchestrator".format(self.__class__.__name__))

        self._settings = read_yaml(yaml_file=kwargs['config'], yaml_type='whatif_config')

        super().__init__(config_folder=kwargs['config_folder'],
                         output_folder=kwargs['output_folder'],
                         exp_id_folder=self._mode + '/' + self._settings['destination_folder'],
                         assets_file=kwargs['assets'],
                         models=kwargs['models'],
                         save_results=kwargs['save_results'],
                         make_plots=kwargs['plot'],
                         )

        self._events = []
        self._timestep = 1
        self._elapsed_time = 0

        # TODO: understand if DONE mi serve
        self.done = False

        # Validate battery parameters unit
        self._settings['battery']['params'] = validate_parameters_unit(self._settings['battery']['params'])

        # Instantiate the BESS environment
        self._battery = BatteryEnergyStorageSystem(
            models_config_files=self._models_configs,
            battery_options=self._settings['battery'],
            input_var='current'
        )

        self._schedule = Schedule(instructions=self._settings['schedule'],
                                  c_value=self._settings['battery']['params']['nominal_capacity'])

    def run(self):
        """

        """
        logger.info("'What-If Simulation' started...")
        self._battery.reset_data()
        self._battery.simulation_init()

        pbar = tqdm(total=len(self._settings['schedule']), position=0, leave=True)
        while not self._schedule.is_empty():
            cmd = self._schedule.get_cmd()
            event_start = self._elapsed_time
            logger.info("Starting command: " + cmd['sentence'])

            if 'duration' in cmd and 'until_cond' not in cmd:
                self._run_for_time(load=list(cmd['load'].keys())[0],
                                   value=list(cmd['load'].values())[0],
                                   time=cmd['duration']
                                   )

            elif 'until_cond' in cmd and 'duration' not in cmd:
                self._run_until_cond(load=list(cmd['load'].keys())[0],
                                     value=list(cmd['load'].values())[0],
                                     cond_var=list(cmd['until_cond'].keys())[0],
                                     cond_value=list(cmd['until_cond'].values())[0],
                                     action=cmd['action']
                                     )

            elif 'until_cond' in cmd and 'duration' in cmd:
                self._run_for_time_or_cond(load=list(cmd['load'].keys())[0],
                                           value=list(cmd['load'].values())[0],
                                           cond_var=list(cmd['until_cond'].keys())[0],
                                           cond_value=list(cmd['until_cond'].values())[0],
                                           time=cmd['duration'],
                                           action=cmd['action']
                                           )

            else:
                logging.error("The experiment configuration is not feasible or not implemented yet")
                exit(1)

            pbar.update(1)
            logger.info("Command executed!")
            self._schedule.next_cmd()
            self._events.append([event_start, self._elapsed_time])

        logger.info("'What-If Simulation' ended without errors!")
        pbar.close()

        self.done = True
        self._output_results(results=self._battery.build_results_table(), summary=self._get_summary())
        if self._make_plots:
            self._prepare_plots()

    def _run_for_time(self, load: str, value: float, time: float):
        """

        Args:
            load ():
            value ():
            time ():
        """
        duration = self._elapsed_time + time
        self._battery.load_var = load
        k = len(self._battery.t_series)

        while self._elapsed_time < duration:
            self._battery.simulation_step(load=value, dt=self._timestep, k=k)
            self._elapsed_time += self._timestep
            self._battery.t_series.append(self._elapsed_time)
            k += 1

    def _run_until_cond(self, load: str, value: float, cond_var: str, cond_value: float, action: str):
        """

        Args:
            load ():
            value ():
            cond_var ():
            cond_value ():
        """
        self._battery.load_var = load
        k = len(self._battery.t_series)

        curr_value = self._battery.get_i if cond_var == 'current' else self._battery.get_v
        op = operator.lt if action == 'charge' else operator.gt

        while op(curr_value(), cond_value):
            self._battery.simulation_step(load=value, dt=self._timestep, k=k)
            self._elapsed_time += self._timestep
            self._battery.t_series.append(self._elapsed_time)
            k += 1

    def _run_for_time_or_cond(self, load: str, value: float, cond_var: str, cond_value: float, time: float, action: str):
        """

        Args:
            load ():
            value ():
            cond_var ():
            cond_value ():
            time ():
        """
        duration = self._elapsed_time + time
        self._battery.load_var = load
        k = len(self._battery.t_series)

        curr_value = self._battery.get_i if cond_var == 'current' else self._battery.get_v
        op = operator.lt if action == 'charge' else operator.gt

        while op(curr_value(), cond_value) and self._elapsed_time < duration:
            self._battery.simulation_step(load=value, dt=self._timestep, k=k)
            self._elapsed_time += self._timestep
            self._battery.t_series.append(self._elapsed_time)
            k += 1

    def _get_summary(self):
        """
        Get simulation summary with important information
        TODO: update when will be added new features
        """
        return {'experiment': self._settings['experiment_name'],
                'description': self._settings['description'],
                'goal': self._settings['goal'],
                'time': self._elapsed_time,
                'battery': self._settings['battery']['params'],
                'initial_conditions': self._settings['battery']['init'],
                'models': [model.__class__.__name__ for model in self._battery.models]
                }

    def _prepare_plots(self):
        """

        """
        var_to_plot = ['voltage', 'temperature', 'power', 'current']

        df = self._battery.build_results_table()

        # Save information for each different kind of plot
        plot_dict = {
            'type': "single",
            'df': df.iloc[1:],
            'variables': var_to_plot,
            'x_ax': 'Time',
            'title': "What-If",
            'events': self._events
        }
        self._plot_info.append(plot_dict)
        self._save_plots()

