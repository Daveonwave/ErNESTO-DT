import pandas as pd
import logging
from tqdm.rich import tqdm
import os

from .base_simulator import BaseSimulator
from .driven_sim import DrivenSimulator
from ernesto.digital_twin.orchestrator import DrivenLoader
from ernesto.digital_twin.orchestrator import DataWriter
from ernesto.adaptation import EvolvingClusteringRoutine, PassiveEvolutionRoutine, ClusterShiftRoutine
from ernesto.digital_twin.bess import BatteryEnergyStorageSystem
from ernesto.adaptation.optimizer import Optimizer
from ernesto.adaptation.regime_shift.parameter_space import ParameterSpace
from ernesto.postprocessing.metrics import _mse, _mape, _max_abs_err

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
        logger.info("Instantiated the {} experiment to enable online adaptation for the estimate of model parameters.".format(self.__class__.__name__))

        super().__init__()
        
        # Data loader and writer
        self._loader = data_loader
        self._writer = data_writer
        
        # The simulator of the DT battery
        self._driven_sim = DrivenSimulator(model_config=model_config,
                                           sim_config=sim_config,
                                           data_loader=data_loader,
                                           data_writer=data_writer)

        self._adapter = globals()[sim_config['adaptive_routine']](
            model_config=model_config,
            sim_config=sim_config,
            clusters_folder=clusters_folder,
            render=sim_config.get('render', False),
            **kwargs
        )

    def _init(self):
        """
        Initialize the adaptive simulation.
        """
        logger.info("'Adaptive Simulation' started...")
        self._driven_sim.init()
        #self._init_state = self._driven_sim.battery.get_snapshot()
        self._adapter.reset(**self._driven_sim.battery.get_snapshot())

        self._driven_sim.init_loader()
        self._driven_sim.load_sample()
        
        self._adapter.track_simulation_state(sample=self._driven_sim.sample, 
                                             domain_vars=self._driven_sim.battery.get_snapshot())
        self._driven_sim.battery._electrical_model.params = self._adapter.get_estimated_params()
                          
    def _run(self):
        """
        Run the adaptive simulation for the whole duration.
        """
        dt = self._loader.timestep if self._loader.timestep is not None else 1
        self._pbar = tqdm(total=int(self._loader.duration), position=0, leave=True)
    
        while not self._driven_sim.done:     
            self._adapter.set_init_state(self._driven_sim.battery.get_snapshot())

            # Run the digital twin battery while the adaptation routine is monitoring
            while self._adapter.keep_monitoring() and not self._driven_sim.done:
                # Step the digital twin simulator
                self._driven_sim.step(dt=dt, sample=self._driven_sim.sample, input_var=self._loader.input_var)
                self._pbar.update(dt)
                dt = self._driven_sim.fetch_next()  
                
                if dt != 0:
                    self._adapter.track_simulation_state(sample=self._driven_sim.sample,
                                                         domain_vars=self._driven_sim.battery.get_snapshot())
                    # Set the parameters of the electrical model for the DT
                    self._driven_sim.battery._electrical_model.params = self._adapter.get_estimated_params()
                
            # Perform the optimization and the adaptation
            self._adapter.step()
            self._driven_sim.battery.clear_collections()
        
        self._pbar.close()
        logger.info("'Adaptive Simulation' solved without errors!")

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
        #self._store_sample()
        self._close()
        
    # def _train(self):
    #     """
    #     # TODO: differentiate between training and adaptation
    #     Create the clusters starting from a current profile. This method use the adaptive simulator just to
    #     collect the data and create the clusters, but it does not perform any adaptation.
    #     """
    #     self._init()
    #     ...
        
    #     points = self._optimizer.estimate_cluster(self._init_state, self._input_batch)
    #     self._grid.current_region.cluster.add(points)
            
    
    # def _store_sample(self):
    #     """
    #     Add the ground and simulated data to the writer queues.
    #     """
    #     for region in self._grid._regions:
    #         region.cluster.save(labels=self._param_names)
    
    def clear(self):
        self._battery.clear_collections()
    
    def _close(self):
        # Export time series data
        if hasattr(self._adapter, 'export_data_history'):
            self._adapter.export_data_history(self._writer._output_folder)
        
        self._loader.destroy()
        self._driven_sim.close()
