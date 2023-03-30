import abc
import pint
from typing import Union
from abc import ABCMeta, abstractmethod, ABC

from src.digital_twin.utils import check_data_unit
from src.digital_twin.units import Unit


class GenericModel(metaclass=ABCMeta):
    """

    """
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'reset_model') and
                callable(subclass.reset_model) and
                hasattr(subclass, 'init_model') and
                callable(subclass.init_model) and
                hasattr(subclass, 'load_battery_state') and
                callable(subclass.load_battery_state) or
                NotImplemented)

    @abc.abstractmethod
    def reset_model(self):
        """

        """
        raise NotImplementedError

    @abc.abstractmethod
    def init_model(self):
        """

        """
        raise NotImplementedError

    @abc.abstractmethod
    def load_battery_state(self, temp, soc, soh):
        """

        """
        raise NotImplementedError


class ElectricalModel(GenericModel):
    """

    """
    def __init__(self, units_checker):
        self.units_checker = units_checker

        self._v_load_series = []
        self._i_load_series = []
        self._times = []

    def reset_model(self):
        pass

    def init_model(self):
        pass

    def load_battery_state(self, temp:float, soc:float, soh:float):
        pass

    def build_components(self, components:dict):
        pass

    def get_v_load_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        """
        if k is not None:
            assert type(k) == int, \
                "Cannot retrieve load voltage of Thevenin model at step K, since it has to be an integer"

            if len(self._v_load_series) > k:
                if not self.units_checker:
                    return self._v_load_series[k]
                else:
                    return self._v_load_series[k].magnitude
            else:
                raise IndexError("Load Voltage V of Thevenin model at step K not computed yet")
        return self._v_load_series

    def get_i_load_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        """
        if k is not None:
            assert type(k) == int, \
                "Cannot retrieve load current of Thevenin model at step K, since it has to be an integer"

            if len(self._i_load_series) > k:
                if not self.units_checker:
                    return self._i_load_series[k]
                else:
                    return self._i_load_series[k].magnitude
            else:
                raise IndexError("Load Current I of Thevenin model at step K not computed yet")
        return self._i_load_series

    def update_v_load(self, value: Union[float, pint.Quantity]):
        if self.units_checker:
            self._v_load_series.append(check_data_unit(value, Unit.VOLT))
        else:
            self._v_load_series.append(value)

    def update_i_load(self, value: Union[float, pint.Quantity]):
        if self.units_checker:
            self._i_load_series.append(check_data_unit(value, Unit.AMPERE))
        else:
            self._i_load_series.append(value)

    def update_times(self, value:int):
        if self.units_checker:
            self._times.append(check_data_unit(value, Unit.SECOND))
        else:
            self._times.append(value)


class ThermalModel(GenericModel):
    """

    """
    def reset_model(self):
        pass

    def init_model(self):
        pass

    def load_battery_state(self, temp, soc, soh):
        pass


class DegradationModel(GenericModel):
    """

    """
    def reset_model(self):
        pass

    def init_model(self):
        pass

    def load_battery_state(self, temp, soc, soh):
        pass