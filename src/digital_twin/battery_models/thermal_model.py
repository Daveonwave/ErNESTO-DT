from src.digital_twin.parameters.variables import Scalar
from src.digital_twin.battery_models import ThermalModel
from src.digital_twin.parameters.variables import instantiate_variables


class RCThermal(ThermalModel):
    """
    Pellegrino paper (@reference [paper link])
    """
    def __init__(self, components_settings: dict):
        super().__init__()

        # TODO: r_term e c_term per ora fisse
        self._thermal_resistance, self._thermal_capacity = instantiate_variables(components_settings)

    @property
    def thermal_resistance(self):
        input_vars = {}

        if not isinstance(self._thermal_resistance, Scalar):
            try:
                input_vars = {name: getattr(self, name) for name in self._thermal_resistance.x_names}
            except:
                raise Exception(
                    "Cannot retrieve required input variables to compute thermal resistance!")

        return self._thermal_resistance.get_value(input_vars=input_vars)

    @property
    def thermal_capacity(self):
        input_vars = {}

        if not isinstance(self._thermal_capacity, Scalar):
            try:
                input_vars = {name: getattr(self, name) for name in self._thermal_capacity.x_names}
            except:
                raise Exception(
                    "Cannot retrieve required input variables to compute thermal capacity!")

        return self._thermal_capacity.get_value(input_vars=input_vars)

    def reset_model(self):
        self._temp_series = []

    def init_model(self, **kwargs):
        """
        Initialize the model at timestep t=0 with an initial temperature equal to 2 degC (ambient temperature)
        """
        temp = kwargs['temperature'] if kwargs['temperature'] else 25
        heat = 0  # kwargs['dissipated_heat'] if kwargs['dissipated_heat'] else 0

        self.update_temp(temp)
        self.update_heat(heat)

    def compute_temp(self, q, env_temp, dt, k=-1):
        """
        Compute the current temperature with equation described in the aforementioned paper

        Inputs:
        :param q: power dissipated adopted as a heat fmu_script
        :param env_temp: ambient temperature
        :param dt: delta of time from last update
        :param k: step
        """
        term_1 = q * self.thermal_resistance * dt
        term_2 = (self.get_temp_series(k=k)) * self.thermal_resistance * self.thermal_capacity
        term_3 = env_temp * dt

        return (term_1 + term_2 + term_3) / (self.thermal_resistance * self.thermal_capacity + dt)


class R2CThermal(ThermalModel):
    """
    Scarpelli-Fioriti paper
    TODO: implement this class (which could be too dependent on cell physical factors)
    """
    def __init__(self,
                 components_settings: dict,
                 ):
        super().__init__()

        self._lambda, self._l, self._a_int, self._a_surf, self._h, self._m, self._cp = (
            instantiate_variables(components_settings))

        self._r_cond = self.l / (self._lambda * self._a_int)
        self._r_conv = 1 / (self._a_int * self._h)

    def reset_model(self):
        self._temp_series = []

    def init_model(self, **kwargs):
        """
        Initialize the model at timestep t=0 with an initial temperature equal to 25 degC (ambient temperature)
        """
        temp = kwargs['temperature'] if kwargs['temperature'] else 25
        heat = 0  # kwargs['dissipated_heat'] if kwargs['dissipated_heat'] else 0

        self.update_temp(temp)
        self.update_heat(heat)

    def compute_temp(self, q, env_temp, dt, k=-1):
        pass
