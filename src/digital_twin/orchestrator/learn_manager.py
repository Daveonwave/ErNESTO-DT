import os
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from rich.pretty import pretty_repr

from src.digital_twin.orchestrator.base_manager import GeneralPurposeManager
from src.digital_twin.bess import BatteryEnergyStorageSystem
from src.postprocessing.visualization import plot_compared_data
from src.preprocessing.data_preparation import load_data_from_csv, validate_parameters_unit
from src.preprocessing.schema import read_yaml

logger = logging.getLogger('DT_ernesto')


class LearningManager(GeneralPurposeManager):
    """

    """
    def __init__(self,
                 models,
                 experiment_config,
                 save_results=False,
                 make_plots=False,
                 **data_folders
                 ):
        super().__init__(experiment_config, save_results, make_plots, **data_folders)

        self._mode = "learning"
