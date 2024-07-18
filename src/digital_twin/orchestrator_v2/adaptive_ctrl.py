import pandas as pd
import logging
from tqdm import tqdm

from src.digital_twin.bess import BatteryEnergyStorageSystem
from src.preprocessing.data_preparation import load_data_from_csv, validate_parameters_unit, sync_data_with_step
from src.preprocessing.schema import read_yaml
from src.postprocessing.metrics import compute_metrics

logger = logging.getLogger('DT_ernesto')


class AdaptiveSimulator():
    """
    Handler of the Compared Simulation experiment.
    -----------------------------------------
    The simulator is conceived to be the orchestrator and the brain of the specified experiment.

    From here, all the kinds of data (input, output, config) are delivered to their consumer hubs, the
    environment is instantiated and the instructions related to the simulation mode chosen by the user are provided.
    """
    def __init__(self, **kwargs):
        pass