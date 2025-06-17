from src.digital_twin.battery_models.generic_models import ThermalModel
from src.digital_twin.parameters import Scalar, instantiate_variables


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
        self._dVoc_dT = self._init_components['dVoc_dT']
        self._soc = None

    @property
    def collections_map(self):
        return {'temperature': self.get_temp_series,
                'heat': self.get_heat_series,
                't_amb': self.get_t_amb_series,
                'dVoc_dT': self.get_dVoc_dT_series}
    
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
    def dVoc_dT(self):
        input_vars = {}

        if not isinstance(self._dVoc_dT, Scalar):
            try:
                input_vars = {name: getattr(self, name) for name in self._dVoc_dT.x_names}
            except:
                raise Exception(
                    "Cannot retrieve required input variables to compute entropic coefficient!")

        return self._dVoc_dT.get_value(input_vars=input_vars)
    
    def get_dVoc_dT_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        """
        if k is not None:
            assert type(k) == int, \
                "Cannot retrieve dVoc_dT of thermal model at step K, since it has to be an integer"

            if len(self._dVoc_dT_series) > k:
                return self._dVoc_dT_series[k]
            else:
                raise IndexError("dVoc_dT of thermal model at step K not computed yet")
        return self._dVoc_dT_series

    def reset_model(self, **kwargs):
        self._temp_series = []
        self._heat_series = []
        self._t_amb_series = []
        self._dVoc_dT_series = []

    def init_model(self, **kwargs):
        """
        Initialize the model at timestep t=0 with an initial temperature equal to 25 degC (ambient temperature)
        """
        temp = kwargs['temperature'] if 'temperature' in kwargs else 298.15
        heat = kwargs['dissipated_heat'] if 'dissipated_heat' in kwargs else 0
        t_amb = kwargs['t_amb'] if 't_amb' in kwargs else 298.15
        dVoc_dT = kwargs['dVoc_dT'] if 'dVoc_dT' in kwargs else self.dVoc_dT

        super()._update_temp(temp)
        super()._update_heat(heat)
        super()._update_t_amb(t_amb)
        self._update_dVoc_dT(dVoc_dT)

    def load_battery_state(self, **kwargs):
        self._soc = kwargs['soc']

    def compute_temp(self, q:float, i:float, T_amb:float, dt:float, k=-1, **kwargs):
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
        denominator = self.c_term / dt + 1 / (self.r_cond + self.r_conv) - self.dVoc_dT * i

        t_core = (term_1 + term_2 + q) / denominator

        t_surf = t_core + self.r_cond * (T_amb - t_core) / (self.r_cond + self.r_conv)

        return t_surf
    
    def update(self, **kwargs):
        super().update(**kwargs)
        self._update_dVoc_dT(self.dVoc_dT)
    
    def _update_dVoc_dT(self, value: float):
        self._dVoc_dT_series.append(value)
    

