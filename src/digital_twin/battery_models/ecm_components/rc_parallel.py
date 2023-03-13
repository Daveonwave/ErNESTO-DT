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
        self._nominal_capacity = None
        self._v = None
        #self.n_r = n_r
        #self.n_c = n_c*

        # Calling the super class constructor we are creating also the i_series collection, but, since in the parallel
        # we have two different currents, here we have to create _i_r1_series and _i_c_series.
        self._i_r1_series = []
        self._i_c_series = []
        self._r1_series = []
        self._c_series = []

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

    @property
    def i_r1_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        """
        if k:
            assert type(k) == int, \
                "Cannot retrieve voltage of {} at step K, since it has to be an integer".format(self._name)

            if len(self._i_r1_series) > k:
                return check_data_unit(self._i_r1_series[k], Unit.OHM)
            else:
                raise IndexError("Voltage V of {} at step K not computed yet".format(self._name))
        return self._i_r1_series

    @property
    def i_c_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        """
        if k:
            assert type(k) == int, \
                "Cannot retrieve voltage of {} at step K, since it has to be an integer".format(self._name)

            if len(self._i_c_series) > k:
                return check_data_unit(self._i_c_series[k], Unit.FARADAY)
            else:
                raise IndexError("Voltage V of {} at step K not computed yet".format(self._name))
        return self._i_c_series

    def _update_i_r1_series(self, value:float):
        self._i_r1_series.append(value)

    def _update_i_c_series(self, value:float):
        self._i_c_series.append(value)

    def compute_v(self, v_ocv, v_r0, v):
        """
        Compute the potential of the RC parallel V_r1=V_c, given the electric current the other potentials of
        the circuit. If we don't have those values we can try to use i_r1.

        Inputs:
        :param v_ocv: potential of open circuit
        :param v_r0: voltage of resistor R0
        :param v: driving potential v(t) in input to the circuit
        """
        if None not in (v_ocv, v_r0, v):
            v_rc = v_ocv - v_r0 - v
        else:
            #TODO: here we have to be sure that last value of i_r1 is the current one
            v_rc = self._i_r1_series[-1] * self._resistance.magnitude
        return check_data_unit(v_rc, Unit.VOLT)

    def compute_i_r1(self, v_rc):
        """
        Compute the flowing electric current I_r1, given the voltage of the resistor V_r1.
        The formula for the computation of I_r1 at step k is:

            I_r1[k] = V_r1[k] / R1

        Inputs
        ------
        :param v_rc : voltage of resistor R1
        """
        i_r1 = v_rc / self._resistance.magnitude
        return check_data_unit(i_r1, Unit.AMPERE)

    def compute_i_c(self, dv_c=None, i=None, i_r1=None):
        """
        Compute the flowing electric current I_c, given the derivative of the capacitor voltage dV_c, by means of the
        capacitor characteristic formula: dV_c = I_c / C (with respect to time).
        In alternative, if available we can use the resistor current I_r1 and the circuit current I. In this case, we
        use the Kirchhoff's law at node: I_c = I - I_r1.

        Inputs
        ------
        param dv_c:
        param i:
        param i_r1:
        """
        if dv_c:
            i_c = dv_c * self._capacity.magnitude
        elif i and i_r1:
            i_c = i - i_r1
        else:
            raise Exception("Not enough data to compute I_c for element {}".format(self.name))
        return check_data_unit(i_c, Unit.AMPERE)

    def compute_tau(self, k):
        """
        Compute the
        """
        tau = self._resistance.magnitude * self._capacity.magnitude
        return check_data_unit(tau, Unit.SECOND)

    def compute_dv(self, i, v_ocv, v, r0):
        """
        Compute the derivative of dv_c/dt using the backward finite differences approach.
        It doesn't work if we don't have v(t).

        Inputs:
        :param i: current at time t
        :param v_ocv: generator of open circuit voltage
        :param v: driving potential v(t) in input to the circuit
        :param r0: resistance of resistor R0
        """
        dv_c = i/self._capacity.magnitude - (v_ocv - r0 * i - v)/(self._resistance.magnitude * self._capacity.magnitude)
        return check_data_unit(dv_c, Unit.VOLT)

    def update_step_variables(self, v, i_r1, i_c, dt, k):
        """
        Aggiorno le liste delle variabili calcolate
        """
        self.update_v(v)
        self._update_i_r1_series(i_r1)
        self._update_i_c_series(i_c)
        self.update_t(self.t_series(k=k-1) + dt)

    def compute_step_variables(self, k:int, v_ocv=None, v=None, i=None,):
        """
        Do computation and update collections

        QUESTO METODO RICEVE LO STEP K, FA LE COMPUTAZIONI E AGGIORNA LE COLLECTIONS
        """
        pass

        
