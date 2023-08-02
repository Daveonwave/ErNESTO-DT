from typing import Union
from src.digital_twin.parameters.variables import Scalar, ParametricFunction, LookupTableFunction
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
    def __init__(self,
                 name,
                 resistance: Union[Scalar, ParametricFunction, LookupTableFunction],
                 capacity: Union[Scalar, ParametricFunction, LookupTableFunction],
                 ):
        super().__init__(name)
        self._resistance = resistance
        self._capacity = capacity
        self._tau = 0
        # TODO: capire tau se trattarla o meno e come trattarla + cambiare unit

        # self.n_r = n_r
        # self.n_c = n_c

        # Collections
        self._i_r1_series = []
        self._i_c_series = []
        self._r1_series = []
        self._c_series = []
        self._tau_series = []

    @property
    def resistance(self):
        """
        Getter of the R1 value. Depending on the x_names (inputs of the function), we retrieve components attribute
        among {SoC, SoH, Temp}.
        If R1 is a scalar, we don't need to provide any input.
        """
        input_vars = {}

        if not isinstance(self._resistance, Scalar):
            try:
                input_vars = {name: getattr(self, name) for name in self._resistance.x_names}
            except:
                raise Exception(
                    "Cannot retrieve required input variables to compute resistance for {}!".format(self.name))

        return self._resistance.get_value(input_vars=input_vars)

    @property
    def capacity(self):
        """
        Getter of the C value. Depending on the x_names (inputs of the function), we retrieve components attribute
        among {SoC, SoH, Temp}.
        If C is a scalar, we don't need to provide any input.
        """
        input_vars = {}

        if not isinstance(self._capacity, Scalar):
            try:
                input_vars = {name: getattr(self, name) for name in self._capacity.x_names}
            except:
                raise Exception(
                    "Cannot retrieve required input variables to compute capacity for {}!".format(self.name))

        return self._capacity.get_value(input_vars=input_vars)

    # TODO: change setter methods -> cannot be done like this with Variables class
    """
    @resistance.setter
    def resistance(self, value: float):
       self._resistance = value

    @capacity.setter
    def capacity(self, value: float):
        self._capacity = value
    """

    def get_r1_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        """
        if k is not None:
            assert type(k) == int, \
                "Cannot retrieve resistance R1 of {} at step K, since it has to be an integer".format(self._name)

            if len(self._r1_series) > k:
                return self._r1_series[k]
            else:
                raise IndexError("Resistance R1 of {} at step K not computed yet".format(self._name))
        return self._r1_series

    def get_c_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        """
        if k is not None:
            assert type(k) == int, \
                "Cannot retrieve capacity C of {} at step K, since it has to be an integer".format(self._name)

            if len(self._c_series) > k:
                return self._c_series[k]
            else:
                raise IndexError("Capacity C of {} at step K not computed yet".format(self._name))
        return self._c_series

    def get_i_r1_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        """
        if k is not None:
            assert type(k) == int, \
                "Cannot retrieve current of {} at step K, since it has to be an integer".format(self._name)

            if len(self._i_r1_series) > k:
                return self._i_r1_series[k]
            else:
                raise IndexError("Current I_r1 of {} at step K not computed yet".format(self._name))
        return self._i_r1_series

    def get_i_c_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        """
        if k is not None:
            assert type(k) == int, \
                "Cannot retrieve current of {} at step K, since it has to be an integer".format(self._name)

            if len(self._i_c_series) > k:
                return self._i_c_series[k]
            else:
                raise IndexError("Current I_c of {} at step K not computed yet".format(self._name))
        return self._i_c_series

    def _update_r1_series(self, value: float):
        self._r1_series.append(value)

    def _update_c_series(self, value: float):
        self._c_series.append(value)

    def _update_i_r1_series(self, value: float):
        self._i_r1_series.append(value)

    def _update_i_c_series(self, value: float):
        self._i_c_series.append(value)

    def init_component(self, r1=0, c=0, i_c=0, i_r1=0):
        """
        Initialize RC component at t=0
        """
        super().init_component()
        self._update_r1_series(r1)
        self._update_c_series(c)
        self._update_i_c_series(i_c)
        self._update_i_r1_series(i_r1)

    def compute_v(self, v_ocv, v_r0, v, i_r1=None):
        """
        Compute the potential of the RC parallel V_r1=V_c, given the electric current the other potentials of
        the circuit. If we don't have those values we can try to use i_r1.

        Inputs:
        :param v_ocv: potential of open circuit
        :param v_r0: voltage of resistor R0
        :param v: driving potential v(t) in input to the circuit
        :param i_r1: current I_r1 flowing through the resistor
        """
        if None not in (v_ocv, v_r0, v):
            v_rc = v_ocv - v_r0 - v
        else:
            v_rc = i_r1 * self.resistance
        return v_rc

    def compute_i_r1(self, v_rc):
        """
        Compute the flowing electric current I_r1, given the voltage of the resistor V_r1.
        The formula for the computation of I_r1 at step k is:

            I_r1[k] = V_r1[k] / R1

        Inputs
        ------
        :param v_rc : voltage of resistor R1
        """
        i_r1 = v_rc / self.resistance
        return i_r1

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
        if dv_c is not None:
            i_c = dv_c * self.capacity
        elif i is not None and i_r1 is not None:
            i_c = i - i_r1
        else:
            raise Exception("Not enough data to compute I_c for element {}".format(self.name))

        return i_c

    def compute_tau(self):
        """
        Compute the
        """
        tau = self.resistance * self.capacity
        return tau

    def update_step_variables(self, r1, c, v_rc, i_r1, i_c, dt:float, k:int):
        """
        Aggiorno le liste delle variabili calcolate
        """
        self._update_r1_series(r1)
        self._update_c_series(c)
        self.update_v(v_rc)
        self._update_i_r1_series(i_r1)
        self._update_i_c_series(i_c)
        #self.update_t(self.get_t_series(k=k - 1) + dt)

