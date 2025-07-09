from typing import Union
from ernesto.digital_twin.parameters.variables import Scalar, ParametricFunction, LookupTableFunction
from ernesto.digital_twin.battery_models.electrical.ecm_components.generic_component import ECMComponent


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
        self._i_r_series = []
        self._i_c_series = []
        self._r_series = []
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

    @resistance.setter
    def resistance(self, new_value):
        self._resistance.set_value(new_value)

    @capacity.setter
    def capacity(self, new_value):
        self._capacity.set_value(new_value)

    def get_r_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        """
        if k is not None:
            assert type(k) == int, \
                "Cannot retrieve resistance of {} at step K, since it has to be an integer".format(self._name)

            if len(self._r_series) > k:
                return self._r_series[k]
            else:
                raise IndexError("Resistance of {} at step K not computed yet".format(self._name))
        return self._r_series

    def get_c_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        """
        if k is not None:
            assert type(k) == int, \
                "Cannot retrieve capacity of {} at step K, since it has to be an integer".format(self._name)

            if len(self._c_series) > k:
                return self._c_series[k]
            else:
                raise IndexError("Capacity of {} at step K not computed yet".format(self._name))
        return self._c_series

    def get_i_r_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        """
        if k is not None:
            assert type(k) == int, \
                "Cannot retrieve curren I_r of {} at step K, since it has to be an integer".format(self._name)

            if len(self._i_r_series) > k:
                return self._i_r_series[k]
            else:
                raise IndexError("Current I_r of {} at step K not computed yet".format(self._name))
        return self._i_r_series

    def get_i_c_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        """
        if k is not None:
            assert type(k) == int, \
                "Cannot retrieve current I_c of {} at step K, since it has to be an integer".format(self._name)

            if len(self._i_c_series) > k:
                return self._i_c_series[k]
            else:
                raise IndexError("Current I_c of {} at step K not computed yet".format(self._name))
        return self._i_c_series

    def reset_data(self):
        self._v_series = []
        self._i_r_series = []
        self._i_c_series = []
        self._r_series = []
        self._c_series = []
        self._tau_series = []

    def init_component(self, r=None, c=None, i_c=0, i_r=0, v_rc=None):
        """
        Initialize RC component at t=0
        """
        r = self.resistance if r is None else r
        c = self.capacity if c is None else c
        v_rc = 0 if v_rc is None else v_rc

        super().init_component(v=v_rc)
        self._update_r_series(r)
        self._update_c_series(c)
        self._update_i_c_series(i_c)
        self._update_i_r_series(i_r)

    def compute_v(self, i_r):
        """
        Compute the potential of the RC parallel V_r1=V_c.

        Inputs:
        :param i_r1: current I_r1 flowing through the resistor
        """
        v_rc = i_r * self.resistance
        return v_rc

    def compute_i_r(self, v_rc):
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

    def compute_i_c(self, dv_c=None, i=None, i_r=None):
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
        elif i is not None and i_r is not None:
            i_c = i - i_r
        else:
            raise Exception("Not enough preprocessing to compute I_c for element {}".format(self.name))

        return i_c

    def compute_tau(self):
        """
        Compute the derivative time of the RC parallel circuit, given the resistance R1 and the capacity C.
        """
        tau = self.resistance * self.capacity
        return tau

    def _update_r_series(self, value: float):
        self._r_series.append(value)

    def _update_c_series(self, value: float):
        self._c_series.append(value)

    def _update_i_r_series(self, value: float):
        self._i_r_series.append(value)

    def _update_i_c_series(self, value: float):
        self._i_c_series.append(value)

    def update_step_variables(self, r, c, v_rc, i_r, i_c):
        """
        Aggiorno le liste delle variabili calcolate
        """
        self._update_r_series(r)
        self._update_c_series(c)
        self.update_v(v_rc)
        self._update_i_r_series(i_r)
        self._update_i_c_series(i_c)

    def clear_collections(self, **kwargs):
        """
        Clear data collections of the component
        """
        self._i_r_series = [self._i_r_series[-1]]
        self._i_c_series = [self._i_c_series[-1]]
        self._r_series = [self._r_series[-1]]
        self._c_series = [self._c_series[-1]]
        super().clear_collections(**kwargs)