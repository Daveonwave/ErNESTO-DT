import os
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from rich.pretty import pretty_repr

from src.digital_twin.handlers.base_manager import GeneralPurposeManager
from src.digital_twin.bess import BatteryEnergyStorageSystem
from src.visualization.fast_plots import plot_compared_data
from src.preprocessing.data_preparation import load_data_from_csv, validate_parameters_unit
from src.preprocessing.schema import read_yaml

logger = logging.getLogger('DT_logger')

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

        self._mode = "whatif"

        # Custom number of iterations or given by preprocessing
        self._iterations = None
        if time_options['iterations']:
            self._iterations = time_options['iterations']
        self._timestep = time_options['timestep']
