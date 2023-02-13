from src.digital_twin.utils import check_data_unit
from src.digital_twin.units import Unit
from src.digital_twin.battery_models.ecm_components.generic_component import ECMComponent


class ResistorCapacitorParallel(ECMComponent):
    """
    Parallel Resistor-Capacitor (RC) element for Thevenin equivalent circuits.

    Parameters
    ----------
    :param resistance:
    :type resistance:

    :param capacity:
    :type capacity:
    """
    def __init__(self, name, resistance, capacity, n_r, n_c):
        super().__init__(name)
        self._resistance = check_data_unit(resistance, Unit.OHM)
        self._capacity = check_data_unit(capacity, Unit.FARADAY)
        self.nominal_capacity = None
        #self.n_r = n_r
        #self.n_c = n_c*

    @property
    def resistance(self):
        return self._resistance

    @resistance.setter
    def resistance(self, value):
        self._resistance = check_data_unit(value, Unit.OHM)

    @property
    def capacity(self):
        return self._capacity

    @capacity.setter
    def capacity(self, value):
        self._capacity = check_data_unit(value, Unit.FARADAY)

    def compute_potential(self, i_r1):
        """
        Compute the potential of the RC parallel V_r1=v_c1, given the electric current i_r1
        """
        v_r1 = check_data_unit(i_r1, Unit.AMPERE).magnitude * self._resistance.magnitude
        return check_data_unit(v_r1, Unit.VOLT)

    def compute_current(self, delta_t, i):
        """
        Compute the flowing electric current I_r1, given the interval of time and the circuit current I
        The formula for the computation of I_r1 at step k is:

            i_R1[k] = exp(-delta_t/(r1c1)) * I_r1[k-1] + (1 - exp(-delta_t/(r1c1))) * I[k-1]

        Inputs
        ------
        :param delta_t: time interval to get the convergence to the steady-state (~5τ)
        :param i: current I flowing in the circuit
        """
        
