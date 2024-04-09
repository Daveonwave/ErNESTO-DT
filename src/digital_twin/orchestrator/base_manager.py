import os
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from rich.pretty import pretty_repr

from src.preprocessing.schema import read_yaml
from src.postprocessing.visualization import plot_compared_data, plot_separate_vars

logger = logging.getLogger('DT_ernesto')


class GeneralPurposeManager:
    """
    Generic handler of the Digital Twin experiment.
    -----------------------------------------
    The simulator is conceived to be the orchestrator and the brain of the specified experiment.

    From here, all the kinds of data (input, output, config) are delivered to their consumer hubs, the
    environment is instantiated and the instructions related to the simulation mode chosen by the user are provided.
    """
    @classmethod
    def get_instance(cls, mode: str):
        """
        Get the instance of the subclass for the current experiment mode, checking if the mode name is
        contained inside the subclass name.
        NOTE: this works because of the __init__.py, otherwise the method __subclasses__() cannot find
              subclasses in other not yet loaded modules.
        """
        logger.info("Starting '{}' experiment...".format(mode))
        return next(c for c in cls.__subclasses__() if mode in c.__name__.lower())

    def __init__(self,
                 config_folder: str,
                 output_folder: str,
                 exp_id_folder: str,
                 assets_file: str,
                 models: list,
                 ground_folder: str = None,
                 save_results: bool = None,
                 save_metrics: bool = None,
                 make_plots: bool = None,
                 ):
        """
        Args:
            config_folder (str):
            output_folder (str):
            ground_folder (str):
            exp_id_folder (str):
            assets_file (str):
            models (list): models to instantiate for the current experiment, given in input by the user
            save_metrics (bool):
            save_results (bool):
            make_plots (bool):
        """
        # Store paths for all different kind of preprocessing
        self._config_folder = Path(config_folder)
        self._assets = read_yaml(yaml_file=assets_file, yaml_type="assets")
        self._output_folder = Path(output_folder) / exp_id_folder / str(datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))
        if ground_folder:
            self._ground_folder = Path(ground_folder)

        # Get models config files
        self._models_configs = []
        for model in models:
            model_file = (Path(self._assets['models_path']) / self._assets['models'][model]['category'] /
                          self._assets['models'][model]['file'])
            self._models_configs.append(read_yaml(yaml_file=self._config_folder / model_file, yaml_type='model'))

        # Output results and postprocessing
        self._save_results = save_results
        self._save_metrics = save_metrics
        self._make_plots = make_plots

        # List of dictionaries with plot information
        self._plot_info = []
        self._results = None

    def run(self):
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def _save_plots(self):
        """
        Save plots in png format inside the output folder in directory img.
        Depending on the required plot, set in the configuration file, it calls the suitable plotter.

        _plot_info structure:
            - compared: {type, dfs, variables, x_axes, labels, title, colors=None}
            - single: {type, df, variables, x_var, title, colors=None}
            - whatif: {type, df, variables, events, x_var, title, colors=None}
        """
        plot_folder = Path(self._output_folder / 'img')
        try:
            os.makedirs(plot_folder, exist_ok=True)
        except NotADirectoryError as e:
            logger.error("It's not possible to create directory {}: {}".format(self._output_folder, e.args))

        for info in self._plot_info:
            if info['type'] == "compared":
                info.pop('type')
                plot_compared_data(dest=plot_folder, **info)

            elif info['type'] == "single":
                info.pop('type')
                plot_separate_vars(dest=plot_folder, **info)

            else:
                raise NotImplementedError("Plot type not implemented yet!")

    def _output_metrics(self, res: dict):
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

    def _output_results(self, results: dict, summary: dict):
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





