import numpy as np
from typing import Union
import pint

from src.digital_twin.battery_models.ecm_components.generic_component import ECMComponent
from src.digital_twin.parameters.variables import Scalar, ParametricFunction, LookupTableFunction
from src.digital_twin.parameters.units import Unit
from src.digital_twin.parameters.data_checker import craft_data_unit


class OCVGenerator(ECMComponent):
    """
    Open Circuit Voltage (OCV) element for Thevenin equivalent circuits.

    Parameters
    ----------

    """
    def __init__(self,
                 name,
                 ocv_potential: Union[Scalar, ParametricFunction, LookupTableFunction],
                 units_checker:bool
                 ):
        super().__init__(name, units_checker)
        self._ocv_potential = ocv_potential
        self._v_ocv_unit = Unit.VOLT

    @property
    def ocv_potential(self):
        input_vars = {}

        if not isinstance(self._ocv_potential, Scalar):
            try:
                input_vars = {name: getattr(self, name) for name in self._ocv_potential.x_names}
            except:
                raise Exception(
                    "Cannot retrieve required input variables to compute resistance for {}!".format(self.name))

        return self._ocv_potential.get_value(input_vars=input_vars)

    @ocv_potential.setter
    def ocv_potential(self, value: Union[float, pint.Quantity]):
        if self.units_checker:
            self._ocv_potential = craft_data_unit(value, Unit.VOLT)
        else:
            self._ocv_potential = value

    def compute_v(self):
        """

        """
        v_ocv = 3.43 + 0.68 * self._soc - 0.68 * (self._soc ** 2) + 0.81 * (self._soc ** 3) - 0.31 * np.exp(-46 * self._soc)

        if self.units_checker:
            v_ocv = craft_data_unit(v_ocv, Unit.VOLT)

        return v_ocv

