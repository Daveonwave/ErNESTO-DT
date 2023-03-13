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
        self._v = None

        # Collections
        self._r0_series = []

    @property
    def resistance(self):
        return self._resistance

    @resistance.setter
    def resistance(self, value):
        self._resistance = check_data_unit(value, Unit.OHM)

    def compute_v(self, i):
        """
        Compute the resistor potential V_r0, given in input the electric current I=I_r0
        #TODO: we will use 'k' when there will be the decay of resistance
        """
        v_r0 = i * self._resistance.magnitude
        return check_data_unit(v_r0, Unit.VOLT)

    def compute_i(self, v_r0):
        """
        Compute the flowing electric current I_r0=I, given in input the resistor potential V_r0
        """
        i_r0 = v_r0 / self._resistance.magnitude
        return check_data_unit(i_r0, Unit.AMPERE)

    def compute_dv(self, i, i_, dt):
        """
        Compute the derivative of (dv_r0/dt) using the backward finite differences approach.
        We consider (dr0/dt) constant, so we can erase the second term of the derivation by parts.

        Inputs:
        :param i: current at time t
        :param i_: current at previous sampling time t-dt
        :param dt: delta of time
        """
        dv_r0 = (i - i_) / dt * self._resistance.magnitude
        return check_data_unit(dv_r0, Unit.VOLT)

    def compute_decay(self, temp, soc, soh):
        """
        # TODO: Update the resistance wrt the degradation of the resistor

        QUESTO METODO RICEVE LO STEP K, FA LE COMPUTAZIONI E AGGIORNA LE COLLECTIONS
        """
        pass

    def update_step_variables(self, v, dt, k):
        """
        Aggiorno le liste delle variabili calcolate
        """
        self.update_v(v)
        self.update_t(self.t_series(k=k-1) + dt)


if __name__ == '__main__':
    resistor = Resistor('name', 5)
    curr = 10 * Unit.AMPERE
    print(resistor.compute_v(curr))