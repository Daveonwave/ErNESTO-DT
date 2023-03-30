import yaml
from pathlib import Path
from src.digital_twin.bess import BatteryEnergyStorageSystem


class Simulator:
    """

    """
    def __init__(self, data_folder, simulation_config, models):
        self.data_folder = Path(data_folder)

        # TODO: unpack also all other settings, like duration, sampling_time (step_frequency) => SIMULATION TYPE
        with open(self.data_folder / Path(simulation_config), 'r') as fin:
            self.simulation_config = yaml.safe_load(fin)

        # TODO: check multiple models of the same type => PUT WARNING OR EXCEPTION
        models_config_files = []
        for model in models:
            model_file = Path(self.simulation_config['models'][model]['category'] + '/' +
                              self.simulation_config['models'][model]['file'])
            models_config_files.append(self.data_folder / model_file)

        self.battery = BatteryEnergyStorageSystem(
            duration=self.simulation_config['duration'],
            sampling_time=self.simulation_config['sampling_time'],
            nominal_capacity=self.simulation_config['nominal_capacity'],
            v_max=self.simulation_config['v_max'],
            v_min=self.simulation_config['v_min'],
            models_config_files=models_config_files,
            units_checker=self.simulation_config['use_data_units']
        )

    def run(self):
        v = self.battery.solve()
        print(v)
