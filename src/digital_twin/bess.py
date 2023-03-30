import yaml

from src.digital_twin.battery_models.electrical_model import TheveninModel
from src.digital_twin.battery_state_manager import SOCEstimator, SOHEstimator
from src.digital_twin.parameters.variables import LookupTableFunction


class BatteryEnergyStorageSystem:
    """
    Class representing the battery abstraction.
    Here we select all the electrical, thermal and mathematical models to simulate the BESS behaviour.
    #TODO: can be done with multi-threading (one for each submodel)?
    """
    def __init__(self,
                 duration=None,
                 sampling_time=None,
                 v_max=None,
                 v_min=None,
                 initial_soc=None,
                 nominal_capacity=None,
                 models=None,
                 units_checker=True,
                 models_config_files=None
                 ):

        self.models_settings = []
        self.units_checker = units_checker

        for file in models_config_files:
            with open(file, 'r') as fin:
                self.models_settings.append(yaml.safe_load(fin))

        print(self.models_settings)

        # Possible models to build
        self.electrical_model = None
        self.thermal_model = None
        self.degradation_model = None
        self.data_driven = None
        self.models = []

        self.build_models()

        # Time options passed by the simulator
        self.duration = duration
        self.sampling_time = sampling_time

        # Electrical options passed by the simulator
        self.nominal_capacity = nominal_capacity
        self.v_max = v_max
        self.v_min = v_min

        self.initial_soc = 0

        # Instantiation of battery state estimators
        self.soc_estimator = SOCEstimator(nominal_capacity=self.nominal_capacity)
        self.soh_estimator = SOHEstimator()

        # Collection where will be stored the simulation variables
        self.soc_series = []
        self.soh_series = []
        self.temp_series = []
        self.t_series = []

    def reset_data(self):
        """

        """
        self.soc_series = []
        self.soh_series = []
        self.temp_series = []
        self.t_series = []

    def build_models(self):
        """
        Model instantiation depending on the 'type' reported in the model yaml file.
        In the same file is annotated also the 'class_name' of the model object to instantiate.

        Accepted 'types' are: ['electrical', 'thermal', 'degradation', 'data_driven'].
        """
        for model_config in self.models_settings:
            if model_config['type'] == 'electrical':
                self.electrical_model = globals()[model_config['class_name']](units_checker=self.units_checker,
                                                                              components=model_config['components'])
                self.models.append(self.electrical_model)

            elif model_config['type'] == 'thermal':
                self.thermal_model = globals()[model_config['class_name']](units_checker=self.units_checker)
                self.electrical_model.build_components(model_config['components'])
                self.models.append(self.thermal_model)

            elif model_config['type'] == 'degradation':
                self.degradation_model = globals()[model_config['class_name']](units_checker=self.units_checker)
                self.electrical_model.build_components(model_config['components'])
                self.models.append(self.degradation_model)

            elif model_config['type'] == 'data_driven':
                self.data_driven = globals()[model_config['class_name']]
                self.electrical_model.build_components(model_config['components'])
                self.models.append(self.data_driven)

            else:
                raise Exception("The 'type' of {} you are trying to instantiate is wrong!"\
                                .format(model_config['class_name']))

    def init(self):
        """
        Initialization of the battery simulation environment at t=0
        """
        # TODO: change this assignation

        for model in self.models:
            model.init_model()

        self.t_series.append(-1)
        self.soc_series.append(self.initial_soc)


    def solve(self):
        """
        #TODO: split in step method
        """
        self.init()

        k = 1
        v_computed = []

        for time in range(0, self.duration, self.sampling_time):
            # Compute the SoC through the SoC estimator and update the state of the circuit
            current_soc = self.soc_estimator.compute_soc(soc_=self.soc_series[k-1], i=-0.2, dt=self.sampling_time)
            self.electrical_model.load_battery_state(soc=current_soc)

            # Solve the circuit for the current step and save the output
            voltage = self.electrical_model.solve_components_cc_mode(dt=self.sampling_time, i_load=-0.2, k=k)
            k += 1
            v_computed.append(voltage)

            self.t_series.append(time)
            self.soc_series.append(current_soc)

            print("I: ", [i for i in self.electrical_model.get_i_load_series()])
            print("V: ", [v for v in self.electrical_model.get_v_load_series()])
            print("TIME: ", self.t_series)

        return v_computed

