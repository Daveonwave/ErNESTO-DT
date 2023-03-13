from src.digital_twin.battery_models.ecm_components.generic_component import ECMComponent
import numpy as np


class OCVGenerator(ECMComponent):
    """
    Open Circuit Voltage (OCV) element for Thevenin equivalent circuits.

    Parameters
    ----------

    """
    def __init__(self, name):
        super().__init__(name)
        self.soc = 0

    @property.setter
    def soc(self, value:float):
        if 0 <= value <= 1:
            self.soc = value
        else:
            raise Exception("The value of the State of Charge (SoC) passed to {} is wrong. "
                            "It has to be comprised between 0 and 1.".format(self.name))

    def compute_v(self, soc=None):
        """

        """
        v = 3.43 + 0.68 * self.soc - 0.68 * (self.soc ** 2) + 0.81 * (self.soc ** 3) - 0.31 * np.exp(-46 * self.soc)
        return v
