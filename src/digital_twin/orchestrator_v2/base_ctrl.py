import os
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.pretty import pretty_repr

from src.preprocessing.schema import read_yaml
from src.postprocessing.visualization import plot_compared_data, plot_separate_vars

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
        logger.info("Instantiated {} class as experiment orchestrator".format(self.__class__.__name__))
        self._settings = read_yaml(yaml_file=kwargs['config'], yaml_type='sim_config')
        
        self.data_loader = None
        self.data_writer = None
        self.controller = None
        
        # Store paths for all different kind of preprocessing
        self._config_folder = Path(kwargs['config_folder'])
        self._assets = read_yaml(yaml_file=kwargs['assets'], yaml_type="assets")
        self._output_folder = Path(kwargs['output_folder']) / self._mode + '/' + self._settings['destination_folder'] / str(datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))
        self._ground_folder = Path(kwargs['ground_folder']) if kwargs['ground_folder'] is not None else None

        # Get models config files
        self._models_configs = []
        for model in kwargs['models']:
            model_file = (Path(self._assets['models_path']) / self._assets['models'][model]['category'] /
                          self._assets['models'][model]['file'])
            self._models_configs.append(read_yaml(yaml_file=self._config_folder / model_file, yaml_type='model'))

        
        
        
        # Output results and postprocessing
        self._results = None
        self._save_results = kwargs['save_results']
        self._save_metrics = kwargs['save_metrics']
    
    def run(self):
        raise NotImplementedError

    def evaluate(self):
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

    def output_results(self, results: dict, summary: dict):
        """

        Args:
            summary (dict):
        """
        if self._save_results:
            try:
                os.makedirs(self._output_folder, exist_ok=True)
            except NotADirectoryError as e:
                logger.error("It's not possible to create directory {}: {}".format(self._output_folder, e.args))

            # Save experiment summary
            with open(self._output_folder / "summary.txt", 'w') as f:
                for key, value in summary.items():
                    f.write('%s: %s\n' % (key, value))

            # Save experiment results
            pd.DataFrame.from_dict(results['operations']).to_csv(self._output_folder / 'dataset.csv', index=False)
            if results['ground']:
                pd.DataFrame.from_dict(results['ground']).to_csv(self._output_folder / 'ground.csv', index=False)
            if results['aging']:
                pd.DataFrame.from_dict(results['aging']).to_csv(self._output_folder / 'aging.csv', index=False)

        # Print on the console the summary and results
        if logger.level == logging.INFO:
            print('\n')
            logger.info("SUMMARY")
            print(pretty_repr(summary))
            logger.info("RESULTS")
            print(str(results))





