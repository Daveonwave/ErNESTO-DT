from src.digital_twin.utils import check_data_unit, craft_data_unit
from src.digital_twin.units import Unit
import pint
from typing import Union

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
    _t_series: collection of all the past discrete steps of time

    Purpose
    -------
    This class builds a generic component of the Thevenin equivalent circuit and presents common attributes and
    collections that can be useful in each single element of the circuit (Resistor, RCParallel, V_OCV generator).
    """
    def __init__(self, name, units_checker=True):
        self._name = name
        self.units_checker = units_checker

        # Dependency variable of components
        #TODO = property methods
        self._temp = None
        self._soc = None
        self._soh = None
        self._k = 0

        # Collections related to the specific component
        self._v_series = []
        self._t_series = []

    @property
    def name(self):
        return self._name

    @property
    def temp(self):
        return self._temp

    @property
    def soc(self):
        return self._soc

    @property
    def soh(self):
        return self._soh

    @property
    def k(self):
        return self._k

    @temp.setter
    def temp(self, value:float):
        self._temp = value

    @soc.setter
    def soc(self, value:float):
        assert 0 <= value <= 1, \
            "The value of the State of Charge (SoC) passed to {} is wrong. " \
            "It has to be comprised between 0 and 1.".format(self.name)
        self._soc = value

    @soh.setter
    def soh(self, value:float):
        assert 0 <= value <= 1, \
            "The value of the State of Health (SoH) passed to {} is wrong. " \
            "It has to be comprised between 0 and 1.".format(self.name)
        self._soh = value

    @k.setter
    def k(self, value:int):
        self._k = value

    def get_v_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        """
        if k is not None:
            assert type(k) == int,\
                "Cannot retrieve voltage of {} at step K, since it has to be an integer".format(self._name)

            if len(self._v_series) > k:
                if not self.units_checker:
                    return self._v_series[k]
                else:
                    return self._v_series[k].magnitude
            else:
                raise IndexError("Voltage V of {} at step K not computed yet".format(self._name))
        return self._v_series

    def get_t_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        #TODO: keep or discard?
        """
        if k is not None:
            assert type(k) == int, \
                "Cannot retrieve elapsed time of {} at step K, since it has to be an integer".format(self._name)

            if len(self._t_series) > k:
                if not self.units_checker:
                    return self._t_series[k]
                else:
                    return self._t_series[k].magnitude
            else:
                raise IndexError("Elapsed time of {} at step K not computed yet".format(self._name))
        return self._t_series

    def update_v(self, value: Union[float, pint.Quantity]):
        if self.units_checker:
            self._v_series.append(check_data_unit(value, Unit.VOLT))
        else:
            self._v_series.append(value)

    def update_t(self, value: Union[float, pint.Quantity]):
        if self.units_checker:
            self._t_series.append(check_data_unit(value, Unit.SECOND))
        else:
            self._t_series.append(value)

    def reset_data(self):
        self._v_series = []
        self._t_series = []

    def init_component(self):
        """

        """
        self.update_v(0)
        self.update_t(0)
