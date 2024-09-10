import pandas as pd
import logging
from tqdm.rich import tqdm

from . import BaseSimulator
from .driven_sim import DrivenSimulator
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
        """
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
        
        self._done = False
        
        # Data loader and writer
        self._loader = data_loader
        self._writer = data_writer
        
        # The simulator of the DT battery
        self._driven_sim = DrivenSimulator(model_config=model_config,
                                          sim_config=sim_config,
                                          data_loader=data_loader,
                                          data_writer=data_writer)
        
        # Instantiate the dual battery for the optimizer
        self._dual_battery = BatteryEnergyStorageSystem(
            models_config=model_config,
            battery_options=sim_config['battery']
        )
        
        optim_info = {'maxiter': 10}
        
        # Adaptive structures
        self._optimizer = Optimizer(battery=self._dual_battery,
                                    alg=kwargs['alg'] if 'alg' in kwargs else sim_config['optimizer']['algorithm'],
                                    alpha=kwargs['alpha'] if 'alpha' in kwargs else sim_config['optimizer']['alpha'],
                                    batch_size=kwargs['batch_size'] if 'batch_size' in kwargs else sim_config['optimizer']['batch_size'],
                                    n_restarts=kwargs['n_restarts'] if 'n_restarts' in kwargs else sim_config['optimizer']['n_restarts'],
                                    bounds=sim_config['optimizer']['search_bounds'],
                                    scale_factor=sim_config['optimizer']['scale_factors'],
                                    **optim_info
                                    )
        self._clusters = None
        self._outliers = None
        
        self._grid = ParameterSpaceGrid(regions=sim_config['adaptation']['regions'])
        
    def init(self):
        """
        Initialize the adaptive simulation.
        """
        logger.info("'Adaptive Simulation' started...")
        self._driven_sim.init()
    
    def run(self):
        pass
        
    def step(self):
        """
        Execute a step of the simulation that can have a fixed or variable timestep.
        
        If the timestep is variable, when the dt overcomes the step size it means that there 
        could be a lack of data in load profile, thus we consider the battery as turned off 
        for (dt-1) seconds by adding a "fake" instruction to rest the battery.
        
        Otherwise, if the timestep is fixed, the simulation progresses with the provided step size.
        
        Args:
            k (int): current iteration of the simulation
            dt (float): delta of time between the current and the previous sample
        """
        
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
        pass
    
    def close(self):
        pass
    