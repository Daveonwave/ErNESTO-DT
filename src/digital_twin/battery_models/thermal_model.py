from src.digital_twin.units import Unit
from src.digital_twin.utils import craft_data_unit, check_data_unit
from src.digital_twin.parameters.variables import Scalar, ParametricFunction, LookupTableFunction
from src.digital_twin.battery_models.generic_models import ThermalModel
from src.digital_twin.parameters.variables import instantiate_variables


class RCThermal(ThermalModel):
    """
    Pellegrino paper (@reference [paper link])
    # TODO: per ora faccio un modello matematico senza oggetti (solo formule)
    """
    def __init__(self,
                 components_settings: dict,
                 units_checker=True
                 ):
        super().__init__(units_checker=units_checker)

        # TODO: r_term e c_term per ora fisse
        self._thermal_resistance, self._thermal_capacity = instantiate_variables(components_settings)
        print(self._thermal_resistance, self._thermal_capacity)

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

    def init_model(self):
#        """ Initialize the model at timestep t=0 with an initial temperature equal to 25°C (ambient temperature)
#        """
        if self.units_checker:
            self.update_temp(craft_data_unit(25, Unit.CELSIUS))
        else:
            self.update_temp(25)

    def compute_temp(self, q, env_temp, dt, k=-1):
        """
        Compute the current temperature with equation described in the aforementioned paper

        Inputs:
        :param q: power dissipated adopted as a heat source
        :param env_temp: ambient temperature
        :param dt: delta of time from last update
        :param k: step
        """
        term_1 = q * self.thermal_resistance * dt
        term_2 = self.get_temp_series(k=k) * self.thermal_resistance * self.thermal_capacity
        term_3 = env_temp * dt

        return (term_1 + term_2 + term_3) / (self.thermal_resistance * self.thermal_capacity + dt)


class R2CThermal(ThermalModel):
    """
    Scarpelli-Fioriti paper
    """
    def __init__(self, units_checher=True):
        super().__init__(units_checker=units_checher)

    def reset_model(self):
        pass

    def init_model(self):
        pass
