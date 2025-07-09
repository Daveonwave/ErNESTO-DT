import pandas as pd
import logging
from tqdm.rich import tqdm

from . import BaseSimulator
from .driven_sim import DrivenSimulator
from ernesto.digital_twin.orchestrator import DrivenLoader
from ernesto.digital_twin.orchestrator import DataWriter
from ernesto.digital_twin.bess import BatteryEnergyStorageSystem
from ernesto.adaptation.optimizer import Optimizer
from ernesto.adaptation.grid import ParameterSpaceGrid

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
                 clusters_folder: str,
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
        self._input_batch = None
        self._batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else sim_config['optimizer']['batch_size']
        self._init_state = {}
        
        # Data loader and writer
        self._loader = data_loader
        self._writer = data_writer
        self._pbar = None
        
        # The simulator of the DT battery
        self._driven_sim = DrivenSimulator(model_config=model_config,
                                           sim_config=sim_config,
                                           data_loader=data_loader,
                                           data_writer=data_writer)
        
        # Instantiate the dual battery for the optimizer
        self._dual_battery = BatteryEnergyStorageSystem(models_config=model_config,
                                                        battery_options=sim_config['battery'])
        
        optim_info = {'maxiter': 10}
        
        # Adaptive structures
        self._optimizer = Optimizer(battery=self._dual_battery,
                                    alg=kwargs['alg'] if 'alg' in kwargs else sim_config['optimizer']['algorithm'],
                                    alpha=kwargs['alpha'] if 'alpha' in kwargs else sim_config['optimizer']['alpha'],
                                    batch_size=kwargs['batch_size'] if 'batch_size' in kwargs else sim_config['optimizer']['batch_size'],
                                    n_restarts=kwargs['n_restarts'] if 'n_restarts' in kwargs else sim_config['optimizer']['n_restarts'],
                                    bounds=sim_config['optimizer']['search_bounds'],
                                    scale_factors=sim_config['optimizer']['scale_factors'],
                                    **optim_info
                                    )
        self._clusters = None
        self._outliers = None
        
        self._grid = ParameterSpaceGrid(grid_config=sim_config['adaptation']['grid'], 
                                        clusters_folder=clusters_folder)
        
    def _add_to_batch(self, sample: dict):
        """
        Add the sample to the batch.
        
        Args:
            sample (dict): The sample to add to the batch
        """
        for key in sample.keys():
            self._input_batch[key].append(sample[key])
            
    def _clear_batch(self):
        """
        Clear the batch.
        """
        self._input_batch = {var:[] for var in self._loader.ground_vars}
        self._input_batch['time'] = []
        
    def _init(self):
        """
        Initialize the adaptive simulation.
        """
        logger.info("'Adaptive Simulation' started...")
        self._driven_sim.init()
        self._init_state = self._driven_sim.battery.get_snapshot()
        
        self._driven_sim.init_loader()
        self._driven_sim.load_sample()
        
        self._clear_batch()
        self._add_to_batch(self._driven_sim.sample)
    
    def _run(self):
        """
        Run the adaptive simulation for the whole duration.
        """
        dt = self._loader.timestep if self._loader.timestep is not None else 1
        #self._pbar = tqdm(total=int(self._loader.duration), position=0, leave=True)
        
        # Check the initial point in the grid
        grid_point = {dim: self._init_state[dim] for dim in self._grid._dimensions}
        self._grid.is_region_changed(point=grid_point)
        # self._driven_sim.battery.update_params(self._grid._regions)
        
        while not self._driven_sim.done:         
            self._init_state = self._driven_sim.battery.get_snapshot()
            
            # Run the digital twin battery for a batch of data
            for _ in tqdm(range(self._batch_size)):
                self._driven_sim.step(dt=dt, sample=self._driven_sim.sample, input_var=self._loader.input_var, timestep=self._loader.timestep)
                self._add_to_batch(self._driven_sim.sample)
                dt = self._driven_sim.fetch_next()
                # dt = self._driven_sim._fetch_next()
            
            # Perform the optimization and the adaptation
            self._step()
            
            #self._pbar.update(self._batch_size)
            self._clear_batch()
            self._driven_sim.battery.clear_collections()
        
    def _step(self):
        """
        Optimization phase: 
        -------------------
        Perform the optimization for the current input batch by estimating the optimal parameters.
        
        Adaptive phase:
        ---------------
        Check if the estimated parameters are within the cluster with an hypothesis test.
        If the hypothesis test fails, the new theta is added to the outlier set.
        Otherwise, the new theta is added to the cluster or nothing happens.
        """
        theta = self._optimizer.step(self._init_state, self._input_batch)
        # self.check_cluster() -> check if the theta is contained within the cluster
        # faulty_cluster_creation() -> create a new cluster with the faulty points
        
        grid_point = {dim: self._init_state[dim] for dim in self._grid._dimensions}
        self._grid.is_region_changed(point=grid_point)
        
        # self._update_params() 
        
        # Call adaptation methods and do stuff with clusters
        # check cluster -> update params with centroids
        
        self._init_state = self._driven_sim.battery.get_snapshot()
        
    def _stop(self):
        """
        Pause the interactive simulation.
        """
        pass
    
    def _solve(self):
        """
        Execute the entire adaptive simulation from the start to the end.
        """
        self._init()
        self._run()
    
    def _store_sample(self):
        """
        Add the ground and simulated data to the writer queues.
        """
        pass
    
    def _close(self):
        pass
    