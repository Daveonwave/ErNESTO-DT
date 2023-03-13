class ECMComponent:
    """
    Generic component of Thevenin equivalent circuits.

    Parameters
    ----------
    :param name: identifier of the component
    :type name: str

    Attributes
    ----------
    _v_series: collection of all the past component voltages
    _i_series: collection of all the past component currents
    _t_series: collection of all the past discrete steps of time

    Purpose
    -------
    This class builds a generic component of the Thevenin equivalent circuit and presents common attributes and
    collections that can be useful in each single element of the circuit (Resistor, RCParallel, V_OCV generator).
    """
    def __init__(self, name):
        self._name = name

        # Collections related to the specific component
        self._v_series = []
        self._t_series = []

    @property
    def name(self):
        return self._name

    @property
    def v_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        """
        if k:
            assert type(k) == int,\
                "Cannot retrieve voltage of {} at step K, since it has to be an integer".format(self._name)

            if len(self._v_series) > k:
                return self._v_series[k]
            else:
                raise IndexError("Voltage V of {} at step K not computed yet".format(self._name))
        return self._v_series

    @property
    def t_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        #TODO: keep or discard?
        """
        if k:
            assert type(k) == int, \
                "Cannot retrieve elapsed time of {} at step K, since it has to be an integer".format(self._name)

            if len(self._t_series) > k:
                return self._t_series[k]
            else:
                raise IndexError("Elapsed time of {} at step K not computed yet".format(self._name))
        return self._t_series

    def update_v(self, value:float):
        self._v_series.append(value)

    def update_t(self, value:int):
        self._t_series.append(value)

    def reset_data(self):
        self._v_series = []
        self._t_series = []

