import yaml
import pandas as pd
from tqdm import tqdm
from time import sleep

from src.data.preprocessing import retrieve_data_from_csv
from src.digital_twin.battery_models.electrical_model import TheveninModel
from src.digital_twin.battery_models.thermal_model import RCThermal
from src.digital_twin.battery_models.aging_model import BolunModel
from src.digital_twin.estimators import SOCEstimator, SOHEstimator
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
                 initial_conditions,
                 ground_file=None,
                 ground_options=None,
                 sign_convention='active',
                 units_checker=True
                 ):

        self.models_settings = []
        self.units_checker = units_checker
        self.sign_convention = sign_convention

        for file in models_config_files:
            with open(file, 'r') as fin:
                self.models_settings.append(yaml.safe_load(fin))

        # Possible electrical to build
        self._electrical_model = None
        self._thermal_model = None
        self._aging_model = None
        self._data_driven = None
        self.models = []

        # Input and ground data preprocessed (TODO: potentially I could want more than one output var (V and power))
        self.load_var = load_options['var']
        self.output_var = ground_options['var']
        self.load_data, self.load_times, _ = retrieve_data_from_csv(csv_file=load_file,
                                                                    var_label=load_options['label'])
        self.ground_data, self.ground_times, _ = retrieve_data_from_csv(csv_file=ground_file,
                                                                        var_label=ground_options['label'])

        # Duration decided in the config file or based on load data timestamps
        if time_options['duration']:
            self.duration = time_options['duration']
        else:
            self.duration = self.load_times[-1] - self.load_times[0]

        # The first element of delta times needs to be added manually, but it must not be 0
        self.delta_times = [1]
        self.delta_times.extend([t - s for s, t in zip(self.load_times, self.load_times[1:])])

        # TODO: understand how to handle sampling time if different from csv data sampling interval => interpolation
        self.sampling_time = time_options['sampling_time']
        self.done = False

        # TODO: both BATTERY OPTIONS and INITIAL COND can depend by the experiment mode => put them in init method
        # Battery options passed by the simulator
        self.nominal_capacity = battery_options['nominal_capacity']
        self.v_max = battery_options['v_max']
        self.v_min = battery_options['v_min']
        self.temp_ambient = battery_options['temp_ambient']

        # Initial battery conditions
        self._initial_conditions = initial_conditions

        # Instantiation of battery state estimators
        self.soc_estimator = SOCEstimator(nominal_capacity=self.nominal_capacity)
        self.soh_estimator = SOHEstimator()

        # Collection where will be stored the simulation variables
        self.soc_series = []
        self.soh_series = []
        # self.temp_series = []
        self.t_series = []
        self.elapsed_time = 0

        # Results -> TODO: probably this is not needed
        self.results = {
            self.output_var: []
        }

        # Instantiate models
        self.build_models()

    def build_models(self):
        """
        Model instantiation depending on the 'type' reported in the model yaml file.
        In the same file is annotated also the 'class_name' of the model object to instantiate.

        Accepted 'types' are: ['electrical', 'thermal', 'degradation', 'data_driven'].
        """
        for model_config in self.models_settings:
            if model_config['type'] == 'electrical':
                self._electrical_model = globals()[model_config['class_name']](units_checker=self.units_checker,
                                                                               components_settings=model_config['components'],
                                                                               sign_convention=self.sign_convention)
                self.models.append(self._electrical_model)

            elif model_config['type'] == 'thermal':
                self._thermal_model = globals()[model_config['class_name']](units_checker=self.units_checker,
                                                                            components_settings=model_config['components'])
                self.models.append(self._thermal_model)

            elif model_config['type'] == 'aging':
                self._aging_model = globals()[model_config['class_name']](units_checker=self.units_checker,
                                                                          components_settings=model_config['components'],
                                                                          stress_models=model_config['stress_models'])
                self.models.append(self._aging_model)

            elif model_config['type'] == 'data_driven':
                self._data_driven = globals()[model_config['class_name']]
                self.models.append(self._data_driven)

            else:
                raise Exception("The 'type' of {} you are trying to instantiate is wrong!"\
                                .format(model_config['class_name']))

    def reset_data(self):
        """

        """
        self.soc_series = []
        self.soh_series = []
        # self.temp_series = []
        self.t_series = []
        self.elapsed_time = 0
        self.done = False

    def build_results_table(self, **kwargs):
        """

        """
        final_dict = {'Time': self.t_series, 'soc': self.soc_series, 'soh': self.soh_series}

        for model in self.models:
            final_dict.update(model.get_final_results())

        for key in final_dict.keys():
            print(key, len(final_dict[key]))

        return pd.DataFrame(data=final_dict, columns=final_dict.keys())

    def save_data(self, output_file, file_name, **kwargs):
        """
        TODO: maybe kwargs can be specific data that have to be specified inside the config file
        """
        try:
            self.build_results_table().to_csv(output_file / file_name, index=False)
        except:
            raise IOError("It is not possible to write the file {}!".format(output_file / file_name))

    """
    ---------------------------------------------
    # Mode: SIMULATION # 
    ---------------------------------------------
    """
    def _simulation_init(self):
        """
        Initialization of the battery simulation environment at t=0
        TODO: maybe some assignements could be added here because they are related to the specific experiment mode
        """
        self.t_series.append(-1)
        self.soc_series.append(self._initial_conditions['soc'])
        self.soh_series.append(self._initial_conditions['soh'])

        for model in self.models:
            model.init_model(**self._initial_conditions)
            model.load_battery_state(temp=self._initial_conditions['temperature'],
                                     soc=self._initial_conditions['soc'],
                                     soh=self._initial_conditions['soh'])

    def _simulation_step(self, dt, k):
        """

        """
        # TODO: calling directly the electrical model is wrong if we have another kind of model, like a data-driven one
        if self.load_var == 'current':
            voltage = self._electrical_model.step_current_driven(i_load=self.load_data[k-1], dt=dt, k=k)

            if voltage == self.v_max or voltage == self.v_min:
                print("VMAX or VMIN REACHED: ", voltage)
                exit()

            i = self.load_data[k-1]
            self.results['voltage'].append(voltage)

        elif self.load_var =='voltage':
            current = self._electrical_model.step_voltage_driven(v_load=self.load_data[k-1], dt=dt, k=k)
            i = current
            self.results['current'].append(current)

        elif self.load_var == 'power':
            voltage, current = self._electrical_model.step_power_driven(p_load=self.load_data[k - 1], dt=dt, k=k)
            i = current
            self.results['voltage'].append(voltage)
            self.results['current'].append(current)

        else:
            raise Exception("The provided battery simulation mode {} doesn't exist or is just not implemented!"
                            "Choose among the provided ones: Voltage, Current or Power.".format(self.load_var))

        # Compute the SoC through the SoC estimator and update the state of the circuit
        dissipated_heat = self._electrical_model.compute_generated_heat()
        curr_temp = self._thermal_model.compute_temp(q=dissipated_heat, env_temp=self.temp_ambient, dt=dt)
        curr_soc = self.soc_estimator.compute_soc(soc_=self.soc_series[-1], i=i, dt=dt)

        # Compute SoH of the system if a model has been selected, SoH=constant otherwise
        if self._aging_model is not None:
            curr_soh = self._aging_model.aging_step()
        else:
            curr_soh = 1

        # Forward SoC, SoH and temperature to models
        for model in self.models:
            model.load_battery_state(temp=curr_temp, soc=curr_soc, soh=curr_soh)

        self._thermal_model.update_temp(value=curr_temp)
        self._thermal_model.update_heat(value=dissipated_heat)
        self.soc_series.append(curr_soc)
        self.soh_series.append(curr_soh)

    def run_simulation(self):
        """
        Run a SIMULATION experiment.
        """
        self._simulation_init()

        k = 1
        pbar = tqdm(total=self.duration, position=0, leave=True)

        # Main loop of the simulation
        while self.elapsed_time < self.duration:

            # dt retrieved by load data
            #print(self.load_times[k-1])
            dt = self.delta_times[k-1]

            # No progress in the simulation
            if dt == 0:
                self.delta_times.pop(k - 1)
                self.load_data.pop(k - 1)
                self.load_times.pop(k - 1)

            else:
                self._simulation_step(dt=dt, k=k)
                self.t_series.append(self.elapsed_time)
                self.elapsed_time += dt
                k += 1

            pbar.update(dt)

        pbar.close()
        self.done = True

    # #################################################################


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

    def _whatif_step(self):
        pass

    def whatif(self):
        pass



