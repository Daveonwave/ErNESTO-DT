from src.digital_twin.units import Unit
from src.digital_twin.utils import check_data_unit
from src.digital_twin.battery_models.ecm_components.generic_component import ECMComponent


class Resistor(ECMComponent):
    """
    Resistor element of Thevenin equivalent circuits.

    Parameters
    ----------
    :param name: identifier of the resistor
    :type name: str

    :param resistance: value of the resistance (Ohm)
    :type resistance: float or int or pint.Quantity

    # TODO: understand how resistance changes and other equations required to reproduce its behaviour
    """
    def __init__(self, name, resistance):
        super().__init__(name)
        self._resistance = check_data_unit(resistance, Unit.OHM)

    @property
    def resistance(self):
        return self._resistance

    @resistance.setter
    def resistance(self, value):
        self._resistance = check_data_unit(value, Unit.OHM)

    def compute_potential(self, i):
        """
        Compute the resistor potential V_r0, given in input the electric current I=I_r0
        """
        v_r0 = check_data_unit(i, Unit.AMPERE).magnitude * self._resistance.magnitude
        return check_data_unit(v_r0, Unit.VOLT)

    def compute_current(self, v_r0):
        """
        Compute the flowing electric current I_r0=I, given in input the resistor potential V_r0
        """
        i_r0 = check_data_unit(v_r0, Unit.VOLT).magnitude / self._resistance.magnitude
        return check_data_unit(i_r0, Unit.AMPERE)

    def compute_decay(self, temp, soc, soh):
        """
        # TODO: Update the resistance wrt the degradation of the resistor
        """
        pass


if __name__ == '__main__':
    resistor = Resistor('name', 5)
    curr = 10 * Unit.AMPERE
    print(resistor.compute_potential(curr))