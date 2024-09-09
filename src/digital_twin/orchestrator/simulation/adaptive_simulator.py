import pandas as pd
import logging
from tqdm import tqdm

from . import BaseSimulator
from src.digital_twin.orchestrator import DrivenLoader
from src.digital_twin.orchestrator import DataWriter
from src.digital_twin.bess import BatteryEnergyStorageSystem
from src.adaptation.optimizer import Optimizer
from src.adaptation.grid import ParameterSpaceGrid

logger = logging.getLogger('ErNESTO-DT')


class AdaptiveSimulator(BaseSimulator):
    """
    Handler of the Compared Simulation experiment.
    -----------------------------------------
    The simulator is conceived to be the orchestrator and the brain of the specified experiment.

    From here, all the kinds of data (input, output, config) are delivered to their consumer hubs, the
    environment is instantiated and the instructions related to the simulation mode chosen by the user are provided.
    """
    def __init__(self, 
                 model_config: dict,
                 sim_config: dict,
                 data_loader: DrivenLoader,
                 data_writer: DataWriter,
                 **kwargs
                 ):
        """_summary_

        Args:
            model_config (dict): _description_
            sim_config (dict): _description_
            data_loader (DrivenLoader): _description_
            data_writer (DataWriter): _description_
        """
        self._mode = "adaptive"
        logger.info("Instantiated the {} experiment to simulate a specific profile.".format(self.__class__.__name__))

        super().__init__()
                
        # Simulation variables
        self._sample = None
        self._get_rest_after = 3600
        self._elapsed_time = 0
        self._done = False
        
        # Data loader and writer
        self._loader = data_loader
        self._writer = data_writer
        
        # Instantiate the BESS environment
        self._battery = BatteryEnergyStorageSystem(
            models_config=model_config,
            battery_options=sim_config['battery']
        )
        
        # Instantiate the dual battery for the optimizer
        self._dual_battery = BatteryEnergyStorageSystem(
            models_config=model_config,
            battery_options=sim_config['battery']
        )
        
        # Adaptive structures
        self._optimizer = Optimizer(battery=self._dual_battery,
                                    alg=kwargs['alg'] if 'alg' in kwargs else sim_config['optimizer']['algorithm'],
                                    alpha=kwargs['alpha'] if 'alpha' in kwargs else sim_config['optimizer']['alpha'],
                                    batch_size=kwargs['batch_size'] if 'batch_size' in kwargs else sim_config['optimizer']['batch_size'],
                                    n_restarts=kwargs['n_restarts'] if 'n_restarts' in kwargs else sim_config['optimizer']['n_restarts'],
                                    bounds=sim_config['optimizer']['search_bounds'],
                                    scale_factor=sim_config['optimizer']['scale_factors']
                                    )
        self._clusters = None
        self._outliers = None
        
        self._grid = ParameterSpaceGrid(regions=sim_config['adaptation']['regions'])
        print("GRId")
        
    def init(self):
        """
        Initialize the adaptive simulation.
        """
        logger.info("'Adaptive Simulation' started...")
        self._battery.reset()
        self._battery.init()
        self._battery.load_var = self._loader.input_var
    
    def run(self):
        """
        
        """
        

    def step(self):
        pass
    
    def stop(self):
        """
        Pause the interactive simulation.
        """
        pass
    
    def solve(self):
        """
        Execute the entire adaptive simulation from the start to the end.
        """
        pass
    
    def store_sample(self):
        """
        Add the ground and simulated data to the writer queues.
        """
        self._writer.add_ground_data(self._sample)
        self._writer.add_simulated_data(self._battery.get_snapshot())
    
    def close(self):
        pass
    