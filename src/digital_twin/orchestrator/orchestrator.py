import os
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.pretty import pretty_repr

from src.preprocessing.schema import read_yaml
from src.preprocessing.data_preparation import validate_parameters_unit
from src.digital_twin.orchestrator import DataLoader
from src.digital_twin.orchestrator import DataWriter
from src.digital_twin.orchestrator.simulation import BaseSimulator

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
        self._settings = read_yaml(yaml_file=kwargs['config'], yaml_type=kwargs['mode'])
        
        # Validate battery parameters unit
        self._settings['battery']['params'] = validate_parameters_unit(self._settings['battery']['params'])
        
        # Store paths for all different kind of preprocessing
        self._config_folder = Path(kwargs['config_folder'])
        self._assets = read_yaml(yaml_file=kwargs['assets'], yaml_type="assets")
        self._output_folder = Path(kwargs['output_folder'] + '/' + kwargs['mode'] + '/' + self._settings['destination_folder'] + '/' + str(datetime.now().strftime('%Y_%m_%d-%H_%M_%S')))
        self._ground_folder = Path(kwargs['ground_folder']) if kwargs['ground_folder'] is not None else None
        if self._ground_folder is not None and kwargs['mode'] != 'scheduled':
            self._settings['input']['ground_data']['file'] = self._ground_folder / self._settings['input']['ground_data']['file']
        
        # Get models config files
        self._models_configs = []
        for model in kwargs['models']:
            model_file = (Path(self._assets['models_path']) / self._assets['models'][model]['category'] /
                          self._assets['models'][model]['file'])
            self._models_configs.append(read_yaml(yaml_file=self._config_folder / model_file, yaml_type='model'))
        
        # Output results and postprocessing
        self._results = None
        
        # Entities useful for the simulation
        self._data_loader = DataLoader.get_instance(mode=kwargs['mode'])(self._settings)
        self._data_writer = DataWriter(output_folder=self._output_folder)
        self._simulator = BaseSimulator.get_instance(mode=kwargs['mode'])(model_config=self._models_configs,
                                                                          sim_config=self._settings,
                                                                          data_loader=self._data_loader,
                                                                          data_writer=self._data_writer)
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
        #TODO: allow interactive simulation from cli to stop/pause/close simulation
        raise NotImplementedError
    
    def output_metrics(self, res: dict):
        """

        Args:
            res ():
        """
        if self._save_metrics:
            try:
                os.makedirs(self._output_folder, exist_ok=True)
            except NotADirectoryError as e:
                logger.error("It's not possible to create directory {}: {}".format(self._output_folder, e.args))

            # Save experiment summary
            df = pd.DataFrame.from_dict(res)
            df.to_csv(self._output_folder / "metrics.csv")

        # Print on the console the summary and results
        else:
            logger.info("METRICS")
            print(pretty_repr(res))
