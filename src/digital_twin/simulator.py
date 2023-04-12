import yaml
from pathlib import Path
from src.digital_twin.bess import BatteryEnergyStorageSystem


class Simulator:
    """
    Simulator of the Digital Twin experiment.
    -----------------------------------------
    The simulator is conceived to be the orchestrator and the brain of the specified simulation/experiment.

    From here, all the kinds of data (input, output, configuration) are delivered to their consumer hubs, the
    environment is instantiated and the instructions related to the simulation mode chosen by the user are provided.
    """
    def __init__(self, mode, simulation_config, models, **data_folders):
        """

        """
        # Store paths for all different kind of data
        self.config_data_path = data_folders['config_data']
        self.load_data_path = data_folders['load_data']
        self.output_data_path = data_folders['output_data']
        self.ground_data_path = data_folders['ground_data']

        # Configure the experiment for the required 'mode'
        self.mode = mode

        # TODO: unpack also all other settings, like duration, sampling_time (step_frequency) => SIMULATION TYPE
        with open(self.config_data_path / Path(simulation_config), 'r') as fin:
            self.simulation_config = yaml.safe_load(fin)

        # TODO: check multiple electrical of the same type => PUT WARNING OR EXCEPTION
        models_config_files = []
        for model in models:
            model_file = Path(self.simulation_config['models'][model]['category'] + '/' +
                              self.simulation_config['models'][model]['file'])
            models_config_files.append(self.config_data_path / model_file)

        # Instantiate the BESS environment
        self.battery = BatteryEnergyStorageSystem(
            models_config_files=models_config_files,
            load_file=self.load_data_path / self.simulation_config['load_csv'],
            load_options=self.simulation_config['load'],
            time_options=self.simulation_config['time'],
            battery_options=self.simulation_config['battery'],
            ground_file=self.ground_data_path / self.simulation_config['ground_csv'],
            ground_options=self.simulation_config['ground'],
            units_checker=self.simulation_config['use_data_units']
        )

    def run(self):
        """

        """
        if self.mode == 'simulation':
            v = self.battery.solve()

            #for v_, ground in zip(v, self.battery.ground_data):
                #print(v_, ground)

            print(len(v), len(self.battery.ground_data))



    def save_results(self):
        """
        # TODO: save results
        """
        results = self.battery.models.get_results()
