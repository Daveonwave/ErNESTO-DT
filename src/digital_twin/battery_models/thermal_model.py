import setuptools

from src.digital_twin.parameters.variables import Scalar
from src.digital_twin.battery_models import ThermalModel
from src.digital_twin.parameters.variables import instantiate_variables


class DummyThermal(ThermalModel):
    """

    """
    def __init__(self, components_settings: None, **kwargs):
        super().__init__(name='dummy_thermal')
        if 'ground_temps' in kwargs:
            self._ground_temps = kwargs['ground_temps']
        else:
            raise AttributeError("The parameter to initialize the attribute of DummyThermal has not benn passed!")

    @property
    def temps(self):
        return self._ground_temps

    def init_model(self, **kwargs):
        """
        Initialize the model at timestep t=0 with an initial temperature equal to 25 degC (ambient temperature)
        """
        temp = kwargs['temperature'] if kwargs['temperature'] else 298.15
        heat = 0  # kwargs['dissipated_heat'] if kwargs['dissipated_heat'] else 0

        self.update_temp(temp)
        self.update_heat(heat)

    def compute_temp(self, **kwargs):
        return self._ground_temps[kwargs['k']]


class R2CThermal(ThermalModel):
    """
    Scarpelli-Fioriti paper
    TODO: implement this class (which could be too dependent on cell physical factors)
    """
    def __init__(self, components_settings: dict, **kwargs):
        super().__init__(name='R2C_thermal')

        self._init_components = instantiate_variables(components_settings)
        self._c_term = self._init_components['c_term']
        self._r_cond = self._init_components['r_cond']
        self._r_conv = self._init_components['r_conv']
        self._dv_dT = self._init_components['dv_dT']
        self._soc = None

    @property
    def soc(self):
        return self._soc

    @property
    def c_term(self):
        input_vars = {}

        if not isinstance(self._c_term, Scalar):
            try:
                input_vars = {name: getattr(self, name) for name in self._c_term.x_names}
            except:
                raise Exception(
                    "Cannot retrieve required input variables to compute thermal capacity!")

        return self._c_term.get_value(input_vars=input_vars)

    @property
    def r_cond(self):
        input_vars = {}

        if not isinstance(self._r_cond, Scalar):
            try:
                input_vars = {name: getattr(self, name) for name in self._r_cond.x_names}
            except:
                raise Exception(
                    "Cannot retrieve required input variables to compute thermal conductive resistor!")

        return self._r_cond.get_value(input_vars=input_vars)

    @property
    def r_conv(self):
        input_vars = {}

        if not isinstance(self._r_conv, Scalar):
            try:
                input_vars = {name: getattr(self, name) for name in self._r_conv.x_names}
            except:
                raise Exception(
                    "Cannot retrieve required input variables to compute thermal convective resistor!")

        return self._r_conv.get_value(input_vars=input_vars)

    @property
    def dv_dT(self):
        input_vars = {}

        if not isinstance(self._dv_dT, Scalar):
            try:
                input_vars = {name: getattr(self, name) for name in self._dv_dT.x_names}
            except:
                raise Exception(
                    "Cannot retrieve required input variables to compute entropic coefficient!")

        return self._dv_dT.get_value(input_vars=input_vars)

    def reset_model(self, **kwargs):
        self._temp_series = []
        self._c_term = self._init_components['c_term']
        self._r_cond = self._init_components['r_cond']
        self._r_conv = self._init_components['r_conv']
        self._dv_dT = self._init_components['dv_dT']

    def init_model(self, **kwargs):
        """
        Initialize the model at timestep t=0 with an initial temperature equal to 25 degC (ambient temperature)
        """
        temp = kwargs['temperature'] if kwargs['temperature'] else 298.15
        heat = 0  # kwargs['dissipated_heat'] if kwargs['dissipated_heat'] else 0

        self.update_temp(temp)
        self.update_heat(heat)

    def load_battery_state(self, **kwargs):
        self._soc = kwargs['soc']

    def compute_temp(self, q, i, T_amb, dt, k=-1):
        """
        Compute the current temperature with equation described in the aforementioned paper
        Args:
            q (float): power dissipated adopted as heat
            i (float): actual current in the circuit
            T_amb (float): ambient temperature
            dt (float): delta of time from last update
            k (int): iteration
        """
        term_1 = self.c_term / dt * self.get_temp_series(k=k)
        term_2 = T_amb / (self.r_cond + self.r_conv)
        denominator = self.c_term / dt + 1 / (self.r_cond + self.r_conv) - self.dv_dT * i

        t_core = (term_1 + term_2 + q) / denominator

        t_surf = t_core + self.r_cond * (T_amb - t_core) / (self.r_cond + self.r_conv)

        return t_surf


class RCThermal(ThermalModel):
    """
    Pellegrino paper (@reference [paper link])
    """
    def __init__(self, components_settings: dict):
        super().__init__(name='RC_thermal')

        # TODO: r_term e c_term per ora fisse
        self._r_term, self._c_term = instantiate_variables(components_settings)

    @property
    def r_term(self):
        input_vars = {}

        if not isinstance(self._r_term, Scalar):
            try:
                input_vars = {name: getattr(self, name) for name in self._r_term.x_names}
            except:
                raise Exception(
                    "Cannot retrieve required input variables to compute thermal resistance!")

        return self._r_term.get_value(input_vars=input_vars)

    @property
    def c_term(self):
        input_vars = {}

        if not isinstance(self._c_term, Scalar):
            try:
                input_vars = {name: getattr(self, name) for name in self._c_term.x_names}
            except:
                raise Exception(
                    "Cannot retrieve required input variables to compute thermal capacity!")

        return self._c_term.get_value(input_vars=input_vars)

    def reset_model(self, **kwargs):
        self._temp_series = []

    def init_model(self, **kwargs):
        """
        Initialize the model at timestep t=0 with an initial temperature equal to 2 degC (ambient temperature)
        """
        temp = kwargs['temperature'] if kwargs['temperature'] else 298.15
        heat = 0  # kwargs['dissipated_heat'] if kwargs['dissipated_heat'] else 0

        self.update_temp(temp)
        self.update_heat(heat)

    def compute_temp(self, q, T_amb, dt, k=-1, i=None):
        """
        Compute the current temperature with equation described in the aforementioned paper

        Inputs:
        :param q: power dissipated adopted as heat
        :param T_amb: ambient temperature
        :param dt: delta of time from last update
        :param k: iteration
        :param i: actual current in the circuit
        """
        term_1 = q * self.r_term * dt
        term_2 = (self.get_temp_series(k=k)) * self.r_term * self.c_term
        term_3 = T_amb * dt

        return (term_1 + term_2 + term_3) / (self.r_term * self.c_term + dt)
