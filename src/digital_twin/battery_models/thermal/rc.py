from src.digital_twin.battery_models.generic_models import ThermalModel
from src.digital_twin.parameters import Scalar, instantiate_variables


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
        temp = kwargs['temperature'] if 'temperature' in kwargs else 298.15
        heat = kwargs['dissipated_heat'] if 'dissipated_heat' in kwargs else 0

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
