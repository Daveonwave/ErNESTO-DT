from pythonfmu import Fmi2Causality, Fmi2Variability, Fmi2Initial, Fmi2Slave, Real

try:
    import os
    import numpy as np
    import yaml
    from pathlib import Path
    from src.digital_twin.bess import BatteryEnergyStorageSystem

except ImportError:
    os, np, yaml, Path, BatteryEnergyStorageSystem = None, None, None, None, None
    print(os, np, yaml, Path, BatteryEnergyStorageSystem)

class DTMockup(Fmi2Slave):
    author = "Davide Salaorni"
    description = "Digital Twin of a Energy Storage System"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        models = ['thevenin', 'rc_thermal']
        config_data_path = "./fmu_script/config"
        experiment_config = "env_config.yaml"

        try:
            with open(config_data_path / Path(experiment_config), 'r') as fin:
                self.experiment_config = yaml.safe_load(fin)
        except Exception:
            raise FileExistsError("Selected config file doesn't exist.")

        models_config_files = []
        for model in models:
            model_file = Path(self.experiment_config['models'][model]['category'] + '/' +
                              self.experiment_config['models'][model]['file'])
            models_config_files.append(config_data_path / model_file)

        self.battery = BatteryEnergyStorageSystem(
            models_config_files=models_config_files,
            battery_options=self.experiment_config['battery'],
            input_var=self.experiment_config['load']['var'],
            output_var=self.experiment_config['ground']['var'],
            sign_convention=self.experiment_config['sign_convention'],
            units_checker=self.experiment_config['use_data_units']
        )

        self.battery.reset_data()
        self.battery.simulation_init(initial_conditions=self.experiment_config['initial_conditions'])

        self.load_current = 0.
        self.output_voltage = 0.
        self.output_current = 0.
        self.k = 1

        self.register_variable(Real("load_current",
                                    causality=Fmi2Causality.input,
                                    variability=Fmi2Variability.continuous)
                               )
        self.register_variable(Real("output_voltage",
                                    causality=Fmi2Causality.output,
                                    variability=Fmi2Variability.continuous,
                                    initial=Fmi2Initial.exact)
                               )
        self.register_variable(Real("k",
                                    causality=Fmi2Causality.output,
                                    variability=Fmi2Variability.discrete,
                                    initial=Fmi2Initial.exact)
                               )

    def do_step(self, current_time, step_size):
        """

        """
        #self.output_current = self.load_current
        self.output_voltage = self.battery.simulation_step(load=self.load_current, dt=step_size, k=self.k)
        self.battery.t_series.append(current_time)
        self.k += 1
        return True
