import yaml
from tqdm import tqdm
from time import sleep

from src.data.preprocessing import retrieve_data_from_csv
from src.digital_twin.battery_models.electrical_model import TheveninModel
from src.digital_twin.battery_models.thermal_model import RCThermal
from src.digital_twin.battery_state_manager import SOCEstimator, SOHEstimator
from src.digital_twin.parameters.variables import LookupTableFunction


class BatteryEnergyStorageSystem:
    """
    Class representing the battery abstraction.
    Here we select all the electrical, thermal and mathematical electrical to simulate the BESS behaviour.
    #TODO: can be done with multi-threading (one for each submodel)?
    """
    def __init__(self,
                 models_config_files,
                 load_file,
                 load_options,
                 time_options,
                 battery_options,
                 ground_file=None,
                 ground_options=None,
                 units_checker=True
                 ):

        self.models_settings = []
        self.units_checker = units_checker

        for file in models_config_files:
            with open(file, 'r') as fin:
                self.models_settings.append(yaml.safe_load(fin))

        # Possible electrical to build
        self.electrical_model = None
        self.thermal_model = None
        self.degradation_model = None
        self.data_driven = None
        self.models = []

        self.build_models()

        # Input and ground data preprocessed
        self.load_data, self.load_times, _ = retrieve_data_from_csv(csv_file=load_file, var_label=load_options['label'])
        self.ground_data, self.ground_times, _ = retrieve_data_from_csv(csv_file=ground_file, var_label=ground_options['label'])

        # Duration decided in the config file or based on load data timestamps
        if time_options['duration']:
            self.duration = time_options['duration']
        else:
            self.duration = self.load_times[-1] - self.load_times[0]

        self.delta_times = [t - s for s, t in zip(self.load_times, self.load_times[1:])]
        # TODO: understand how to handle sampling time
        self.sampling_time = time_options['sampling_time']

        # Battery options passed by the simulator
        self.nominal_capacity = battery_options['nominal_capacity']
        self.v_max = battery_options['v_max']
        self.v_min = battery_options['v_min']
        self.initial_soc = 0

        # Instantiation of battery state estimators
        self.soc_estimator = SOCEstimator(nominal_capacity=self.nominal_capacity)
        self.soh_estimator = SOHEstimator()

        # Collection where will be stored the simulation variables
        self.soc_series = []
        self.soh_series = []
        # self.temp_series = []
        self.t_series = []

    def reset_data(self):
        """

        """
        self.soc_series = []
        self.soh_series = []
        # self.temp_series = []
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
                                                                              components_settings=model_config['components'])
                self.models.append(self.electrical_model)

            elif model_config['type'] == 'thermal':
                self.thermal_model = globals()[model_config['class_name']](units_checker=self.units_checker,
                                                                           components_settings=model_config['components'])
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

    def _simulation_init(self):
        """
        Initialization of the battery simulation environment at t=0
        """
        # TODO: change this assignation

        for model in self.models:
            model.init_model()

        self.t_series.append(-1)
        self.soc_series.append(self.initial_soc)

    def _simualation_step(self, mode:str):
        """

        """

    def solve(self):
        """

        """
        self._simulation_init()

        elapsed_time = 0
        k = 1
        v_computed = []

        pbar = tqdm(total=self.duration, position=0, leave=True)
        while elapsed_time < self.duration:

            dt = self.delta_times[k-1]

            # If dt is equal to 0, we get some troubles in computation, so we need to pop those frames from load collections
            if dt == 0:
                self.delta_times.pop(k-1)
                self.load_data.pop(k-1)
                self.load_times.pop(k-1)

            else:
                # Compute the SoC through the SoC estimator and update the state of the circuit
                current_soc = self.soc_estimator.compute_soc(soc_=self.soc_series[-1], i=self.load_data[k-1], dt=dt)
                self.electrical_model.load_battery_state(soc=current_soc)

                # Solve the circuit for the current step and save the output
                voltage = self.electrical_model.solve_components_cc_mode(dt=dt, i_load=self.load_data[k-1], k=k)

                temp = self.thermal_model.compute_temp(q=self.electrical_model.compute_generated_heat(k=k),
                                                       env_temp=25.0,
                                                       dt=dt)
                print("TEMPERATURE: ", temp)
                self.thermal_model.update_temp(value=temp)

                v_computed.append(voltage)
                self.t_series.append(elapsed_time)
                self.soc_series.append(current_soc)

                #print("I: ", [i for i in self.electrical_model.get_i_load_series()])
                #print("V: ", [v for v in self.electrical_model.get_v_load_series()])
                #print("TIME: ", self.t_series)
                #print("time: ", elapsed_time)

                elapsed_time += dt
                k += 1

            #ssleep(0.01)
            pbar.update(dt)

        pbar.close()
        return v_computed


    """
    LEARNING MODE
    """
    def _learning_init(self):
        pass

    def _learning_step(self):
        pass

    def learn(self):
        pass


    """
    WHAT-IF MODE
    """
    def _whatif_init(self):
        pass

    def whatif_step(self):
        pass

    def whatif(self):
        pass

