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

        # Prepare ground preprocessing for input and validation
        self._input_var = None



        self._schedule = self._parse_schedule()

        # TODO: understand if DONE mi serve
        self.done = False

        # Validate battery parameters unit
        self._settings['battery']['params'] = validate_parameters_unit(self._settings['battery']['params'])

        # Instantiate the BESS environment
        self._battery = BatteryEnergyStorageSystem(
            models_config_files=self._models_configs,
            battery_options=self._settings['battery'],
            input_var=self._input_var
        )

    def _parse_schedule(self):
        pass

    def run(self):
        """
        """


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
                'battery': self._settings['battery']['params'],
                'initial_conditions': self._settings['battery']['init'],
                'models': [model.__class__.__name__ for model in self._battery.models]
                }

    def _prepare_plots(self):
        pass
