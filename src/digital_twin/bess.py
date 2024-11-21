from .battery_models import *


class BatteryEnergyStorageSystem:
    """
    Class representing the battery abstraction.
    Here we select all the electrical, thermal and mathematical electrical to simulate the BESS behaviour.
    #TODO: can be done with multi-threading (one for each submodel)?
    """
    def __init__(self,
                 models_config: list,
                 battery_options: dict,
                 input_var: str='current',
                 check_soh_every=None,
                 **kwargs
                 ):
        """_summary_

        Args:
            models_config (list): _description_
            battery_options (dict): _description_
            input_var (str, optional): _description_. Defaults to 'current'.
            check_soh_every (_type_, optional): _description_. Defaults to None.
        """
        self.models_settings = models_config
        self._load_var = input_var
        self._ground_data = kwargs["ground_data"] if "ground_data" in kwargs else None

        # Possible electrical to build
        self._electrical_model = None
        self._thermal_model = None
        self._aging_model = None
        self._soc_model = None
        self.models = []

        # TODO: both BATTERY OPTIONS and INITIAL COND can depend by the experiment mode => put them in init method
        # Battery options passed by the simulator
        self.nominal_capacity = battery_options['params']['nominal_capacity']
        self.nominal_dod = battery_options['params']['nominal_dod'] \
            if 'nominal_dod' in battery_options['params'].keys() else None
        self.nominal_lifetime = battery_options['params']['nominal_lifetime'] \
            if 'nominal_lifetime' in battery_options['params'].keys() else None
        self.nominal_voltage = battery_options['params']['nominal_voltage'] \
            if 'nominal_voltage' in battery_options['params'].keys() else None
        self._v_max = battery_options['params']['v_max']
        self._v_min = battery_options['params']['v_min']
        #self._temp_ambient = battery_options['params']['temp_ambient']
        
        # Bounds of operating conditions of the battery
        self.soc_min = battery_options['bounds']['soc']['low'] if 'bounds' in battery_options.keys() else 0.
        self.soc_max = battery_options['bounds']['soc']['high'] if 'bounds' in battery_options.keys() else 1.
        
        # Initial conditions of the battery
        self._init_conditions = battery_options['init']
        self._sign_convention = battery_options['sign_convention']
        self._reset_soc_every = battery_options['reset_soc_every'] if 'reset_soc_every' in battery_options['params'].keys() else None
        self._check_soh_every = check_soh_every if check_soh_every is not None else 3600

        # Collection where will be stored the simulation variables
        self.soc_series = []
        self.soh_series = []
        self.t_series = []
        self.c_max_series = []

        # Instantiate models
        self._build_models()

    @property
    def load_var(self):
        return self._load_var

    @load_var.setter
    def load_var(self, var: str):
        self._load_var = var

    def get_v(self):
        return self._electrical_model.get_v_series(k=-1)

    def get_i(self):
        return self._electrical_model.get_i_series(k=-1)

    def get_feasible_current(self, last_soc:float=None, dt:float=1):
        """
        Get the feasible min and max currents that can be applied to the battery at the current time step.
        
        Args:
            last_soc (float, optional): last value of the SoC. Defaults to None.
            dt (float, optional): delta of time. Defaults to 1.
        """
        soc_ = self.soc_series[-1] if last_soc is None else last_soc
        return self._soc_model.get_feasible_current(soc_=soc_, dt=dt)

    def _build_models(self):
        """
        Model instantiation depending on the 'type' reported in the model yaml file.
        In the same file is annotated also the 'class_name' of the model object to instantiate.

        Accepted 'types' are: ['electrical', 'thermal', 'degradation'].
        """
        for model_config in self.models_settings:
            if model_config['type'] == 'electrical':
                self._electrical_model = globals()[model_config['class_name']](components_settings=model_config['components'],
                                                                               sign_convention=self._sign_convention)
                self.models.append(self._electrical_model)

            elif model_config['type'] == 'thermal':
                components = model_config['components'] if 'components' in model_config.keys() else None
                #kwargs = {'ground_temps': self._ground_data['temperature'] if 'temperature' in self._ground_data else None}
                self._thermal_model = globals()[model_config['class_name']](components_settings=components)
                self.models.append(self._thermal_model)

            elif model_config['type'] == 'aging':
                self._aging_model = globals()[model_config['class_name']](components_settings=model_config['components'],
                                                                          stress_models=model_config['stress_models'],
                                                                          init_soc=self._init_conditions['soc'])
                self.models.append(self._aging_model)

            else:
                raise Exception("The 'type' of {} you are trying to instantiate is wrong!"\
                                .format(model_config['class_name']))
        
        # Instantiation of battery state estimators
        self._soc_model = SOCEstimator(capacity=self.nominal_capacity, soc_max=self.soc_max, soc_min=self.soc_min)

    def reset(self, reset_info: dict = {}):
        """
        Reset the battery simulation environment to the initial conditions.
        
        Args:
            reset_info (dict, optional): settings to reset the battery. Defaults to {}.
        """
        self.soc_series = []
        self.soh_series = []
        self.t_series = []
        self.c_max_series = []

        for model in self.models:
            model.reset_model(**reset_info)

    def init(self, init_info: dict = {}):
        """
        Initialize the battery simulation environment.

        Args:
            init_info (dict, optional): settings to initialize the battery. Defaults to {}.
        """
        self.t_series.append(-1)
        self.soc_series.append(self._init_conditions['soc'])
        self.soh_series.append(self._init_conditions['soh'])
        self.c_max_series.append(self.nominal_capacity)

        for model in self.models:
            model.load_battery_state(temp=self._init_conditions['temperature'],
                                     soc=self._init_conditions['soc'],
                                     soh=self._init_conditions['soh'])

            model.init_model(**self._init_conditions)

    def step(self, load: float, dt: float, k: int, t_amb: float = None, ground_temp: float = None):
        """
        Perform a step of the simulation by applying the load to the battery and updating the state of the system.

        Args:
            load (float): value of the load to apply to the battery.
            dt (float): delta of time between the current and the previous sample.
            k (int): k-th iteration of the simulation.
            ground_temp (float, optional): actual ground temperature to consider in the thermal model (if needed).

        Raises:
            Exception: if the provided battery simulation mode doesn't exist or is just not implemented.
        """
        if self._load_var == 'current':
            v_out, _ = self._electrical_model.step_current_driven(i_load=load, dt=dt, k=-1)
            i = load

        elif self._load_var == 'voltage':
            _, i_out = self._electrical_model.step_voltage_driven(v_load=load, dt=dt, k=-1)
            i = i_out
            v_out = load

        elif self._load_var == 'power':
            v_out, i_out = self._electrical_model.step_power_driven(p_load=load, dt=dt, k=-1)
            i = i_out

        else:
            raise Exception("The provided battery simulation mode {} doesn't exist or is just not implemented!"
                            "Choose among the provided ones: Voltage, Current or Power.".format(self._load_var))

        # Compute the SoC through the SoC estimator and update the state of the circuit
        dissipated_heat = self._electrical_model.compute_generated_heat()

        curr_temp = self._thermal_model.compute_temp(q=dissipated_heat, i=i, T_amb=t_amb, dt=dt, k=-1, ground_temp=ground_temp)
        curr_soc = self._soc_model.compute_soc(soc_=self.soc_series[-1], i=i, dt=dt)

        self._thermal_model.update_temp(value=curr_temp)
        self._thermal_model.update_heat(value=dissipated_heat)
        self.soc_series.append(curr_soc)

        # Compute SoH of the system if a model has been selected, SoH=constant otherwise
        curr_soh = self.soh_series[-1]
        if self._aging_model is not None and self._aging_model.name == 'Bolun':
            if k % self._check_soh_every == 0:
                curr_soh = self.soh_series[0] - \
                    self._aging_model.compute_degradation(soc_history=self.soc_series,
                                                          temp_history=self._thermal_model.get_temp_series(),
                                                          elapsed_time=self.t_series[-1],
                                                          k=k)
        
        # BOLUN DROPFLOW MODEL
        if self._aging_model is not None and self._aging_model.name == 'BolunDropflow':
            curr_soh = self.soh_series[0] - \
                self._aging_model.compute_degradation(soc=self.soc_series[-1],
                                                      temp=self._thermal_model.get_temp_series(k=-1),
                                                      elapsed_time=self.t_series[-1],
                                                      k=k,
                                                      do_check=(k % self._check_soh_every == 0))
                        
        self.soh_series.append(curr_soh)
        
        # Update the maximum capacity of the battery and the SoC model since the battery capacity fades with SoH
        curr_c_max = self.nominal_capacity * curr_soh
        self._soc_model.c_max = curr_c_max
        self.c_max_series.append(curr_c_max)

        # Forward SoC, SoH and temperature to models and their components
        for model in self.models:
            model.load_battery_state(temp=curr_temp, soc=curr_soc, soh=curr_soh)

        # Reset the SoC estimation to avoid an error drift of the SoC estimation. TODO: move this in the SoC model maybe?
        if self._reset_soc_every is not None and k % self._reset_soc_every == 0:
            self.soc_series[-1] = self._soc_model.reset_soc(v=v_out, v_max=self._v_max, v_min=self._v_min)

    def get_snapshot(self):
        """
        Collect the status of the battery and its components at the current time step.
        Used to update the queues of the writer.
        """
        status_dict = {'time': self.t_series[-1], 'soc': self.soc_series[-1], 'soh': self.soh_series[-1], 'c_max': self.c_max_series[-1]}

        for model in self.models:
            status_dict.update(model.get_results(**{'k': -1}))

        return status_dict

    def get_collections(self, var_names: list = None):
        """
        Collct results of the entire simulation and return them in a dictionary.
        NOTE: No more used since we have the writer to collect data by queues updated at each step.
        """
        available_vars = {
            'time': self.t_series,
            'soc': self.soc_series,
            'soh': self.soh_series,
            'c_max': self.c_max_series
        }

        # Collect requested variables or all if var_names is None
        battery_dict = {var: available_vars[var] for var in var_names if var in available_vars} if var_names is not None else available_vars.copy()
        info_dict = {'var_names': var_names} if var_names is not None else {}

        for model in self.models:
            battery_dict.update(model.get_results(**info_dict)) 

        return battery_dict
    
    def clear_collections(self):
        """
        Clear the collections of the battery simulation.
        """
        self.soh_series = [self.soh_series[-1]]
        self.t_series = [self.t_series[-1]]
        self.c_max_series = [self.c_max_series[-1]]
        self.soc_series = [self.soc_series[-1]]
        
        for model in self.models:
            model.clear_collections()
            




