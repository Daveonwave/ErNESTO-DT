import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.pretty import pretty_repr

from ernesto.preprocessing.schema import read_yaml
from ernesto.preprocessing.data_preparation import validate_parameters_unit
from ernesto.digital_twin.orchestrator import DataLoader
from ernesto.digital_twin.orchestrator import DataWriter
from ernesto.digital_twin.orchestrator.simulation import BaseSimulator, DrivenSimulator, ScheduledSimulator, AdaptiveSimulator

logger = logging.getLogger('ErNESTO-DT')


class DTOrchestrator:
    """
    Generic orchestrator of the Digital Twin.
    -----------------------------------------
    The orchestrator is conceived to be controller of the specified experiment.

    From here, all the kinds of data (input, output, config) are delivered to their consumer hubs, the
    environment is instantiated and the instructions related to the simulation mode chosen by the user are provided.
    """
    def __init__(self, **kwargs):
        # Entities controlled by the orchestrator
        logger.info("Starting ErNESTO-DT...")
        
        # This allow to read configuration in the main script (this allows to have multiple simulation without
        # creating each single yaml file)
        if 'already_read' in kwargs and kwargs['already_read']:
            self._settings = kwargs['config']
        else:
            self._settings = read_yaml(yaml_file=kwargs['config'], yaml_type=kwargs['mode'])
        
        # Validate battery parameters unit
        self._settings['battery']['params'] = validate_parameters_unit(self._settings['battery']['params'])
        
        # Store paths for all different kind of preprocessing
        self._config_folder = Path(kwargs['config_folder'])
        # self._assets = read_yaml(yaml_file=kwargs['assets'], yaml_type="assets")
        self._output_folder = Path(kwargs['output_folder'] + '/' + kwargs['mode'] + '/' + self._settings['destination_folder'] + '/' + str(datetime.now().strftime('%Y_%m_%d-%H_%M_%S')))
        self._ground_folder = Path(kwargs['ground_folder']) if kwargs['ground_folder'] is not None else None
        if self._ground_folder is not None and kwargs['mode'] != 'scheduled':
            self._settings['input']['ground_data']['file'] = self._ground_folder / self._settings['input']['ground_data']['file']
        
        # Get models config files
        self._models_configs = []
        for key, model in kwargs['models'].items():
            model_file = (Path("./models/{}/{}.yaml".format(key, model)))
            self._models_configs.append(read_yaml(yaml_file=self._config_folder / model_file, yaml_type='model'))
        
        # Validate if models can simulate the chosen experiment
        _types = [config['type'] for config in self._models_configs]
        eletr_check = 'electrical' in _types
        therm_check = 'thermal' in _types
        aging_check = 'aging' in _types
        assert(eletr_check and (not aging_check or therm_check)), \
            "The models selected to run the experiment are not compatible.\n\
                Be sure to have at least the electrical model.\n\
                If you need the aging model, you must have also the thermal one.\n\
                Otherwise, you can run the thermal model without the aging one."
        
        # Output results and postprocessing
        self._results = None
        kwargs['output_folder'] = self._output_folder
        
        # Entities useful for the simulation
        self._data_loader = DataLoader.get_instance(mode=list(self._settings['input'].keys())[0])(self._settings)
        self._data_writer = DataWriter(output_folder=self._output_folder)
        self._simulator = BaseSimulator.get_instance(mode=kwargs['mode'])(model_config=self._models_configs,
                                                                          sim_config=self._settings,
                                                                          data_loader=self._data_loader,
                                                                          data_writer=self._data_writer,
                                                                          **kwargs)
        self._interactive = kwargs['interactive']
    
    def run(self):
        if self._interactive:
            self._run_interactive()
        else:
            self._run()
            
        self._simulator.close()
        logger.info("Shutting down ErNESTO-DT...")
            
    def _run(self):
        """
        Run an unstoppable simulation which doesn't provide a user interface to interact with.
        """
        self._simulator.solve()
    
    def _run_interactive(self):
        #TODO: allow interactive simulation from cli to stop/pause/close simulation with a dedicated thread
        self._simulator.run()
        
    
    



