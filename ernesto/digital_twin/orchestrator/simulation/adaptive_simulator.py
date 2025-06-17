import pandas as pd
import logging
from tqdm.rich import tqdm

from .base_simulator import BaseSimulator
from .driven_sim import DrivenSimulator
from ernesto.digital_twin.orchestrator import DrivenLoader
from ernesto.digital_twin.orchestrator import DataWriter
from ernesto.digital_twin.bess import BatteryEnergyStorageSystem
from ernesto.adaptation.optimizer import Optimizer
from ernesto.adaptation.parameter_grid import ParameterSpaceGrid

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
        
        self._enable_adaptation = kwargs['enable_adaptation']
        self._param_names = sim_config['adaptation']['param_names']
        
        # Simulation variables
        self._input_batch = []
        self._batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else sim_config['optimizer']['batch_size']
        self._init_state = {}
        
        # Data loader and writer
        self._loader = data_loader
        self._writer = data_writer
        
        # The simulator of the DT battery
        self._driven_sim = DrivenSimulator(model_config=model_config,
                                           sim_config=sim_config,
                                           data_loader=data_loader,
                                           data_writer=data_writer)
        
        optim_info = sim_config['optimizer']
        
        # Adaptive structures
        self._optimizer = Optimizer(battery_config={'models_config': model_config, 
                                                    'battery_options': sim_config['battery']},
                                    **optim_info,
                                    enabled_adaptation=self._enable_adaptation,
                                    )
        
        self._grid = ParameterSpaceGrid(grid_config=sim_config['parameters_grid'], 
                                        clusters_folder=clusters_folder,
                                        output_folder=kwargs['output_folder'])
        
        self._region = None
        
        
    def _add_to_batch(self, sample: dict):
        """
        Add the sample to the batch.
        
        Args:
            sample (dict): The sample to add to the batch
        """
        self._input_batch.append(sample)
            
    def _clear_batch(self):
        """
        Clear the batch.
        """
        self._input_batch = []
        
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
                  
    def _run(self):
        """
        Run the adaptive simulation for the whole duration.
        """
        dt = self._loader.timestep if self._loader.timestep is not None else 1
        
        # Check the initial point in the grid
        grid_point = {dim: self._init_state[dim] for dim in self._grid._dimensions}
        self._region = self._grid.check_region(points=[grid_point])
        print("Current region: ", self._region)
        
        # The cluster is initially empty, so we need to add the initial point to the cluster
        if self._region.cluster.is_empty():
            params = self._driven_sim.battery._electrical_model.params
            self._region.cluster.add([params[name] for name in self._param_names])
            print("Added initial point to the cluster: ", params)
        
        # Get the centroid of the cluster and set it as the initial parameters of the battery
        centroid = self._region.cluster.centroid
        self._driven_sim.battery._electrical_model.params = {label: centroid[i] for i, label in enumerate(self._param_names)}
        print("centroid: ", centroid) 
        print("params: ", self._driven_sim.battery._electrical_model.params)
        exit()
        
        if self._batch_size is None:
            self._batch_size = self._loader.duration
        
        while not self._driven_sim.done:     
            self._init_state = self._driven_sim.battery.get_snapshot()
        
            # Run the digital twin battery for a batch of data
            for _ in tqdm(range(self._batch_size)):
                self._driven_sim.sample['voltage'] = self._driven_sim.battery.get_v()
                self._add_to_batch(self._driven_sim.sample)
                self._driven_sim.step(dt=dt, sample=self._driven_sim.sample, input_var=self._loader.input_var)
                dt = self._driven_sim.fetch_next()            
            
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
        if self._enable_adaptation:
            theta = self._optimizer.estimate_new_heta(self._init_state, self._input_batch)
            # self.check_cluster() -> check if the theta is contained within the cluster
            # faulty_cluster_creation() -> create a new cluster with the faulty points
        
            grid_point = {dim: self._init_state[dim] for dim in self._grid._dimensions}
            self._grid.is_region_changed(point=grid_point)
            
            # self._update_params() 
            
            # Call adaptation methods and do stuff with clusters
            # check cluster -> update params with centroids
        else:
            points = self._optimizer.estimate_cluster(self._init_state, self._input_batch)
            
            self._grid.current_region.cluster.add(points)
        
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
        self._store_sample()
        self._close()
    
    def _store_sample(self):
        """
        Add the ground and simulated data to the writer queues.
        """
        for region in self._grid._regions:
            region.cluster.save(labels=self._param_names)
    
    def clear(self):
        self._battery.clear_collections()
    
    def _close(self):
        self._loader.destroy()
        self._driven_sim.close()    