import abc
from abc import ABCMeta


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
                callable(subclass.load_battery_state) and
                hasattr(subclass, 'get_final_results') and
                callable(subclass.get_final_results) or
                NotImplemented)

    @abc.abstractmethod
    def reset_model(self):
        raise NotImplementedError

    @abc.abstractmethod
    def init_model(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def load_battery_state(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def get_results(self, **kwargs):
        raise NotImplementedError
    
    @abc.abstractmethod
    def clear_collections(self, **kwargs):
        raise NotImplementedError


class ElectricalModel(GenericModel):
    """

    """
    def __init__(self, name: str):
        self._name = name
        self._params = []
        self._v_load_series = []
        self._i_load_series = []
        self._power_series = []

    @property
    def name(self):
        return self._name

    @property
    def param_names(self):
        return self._params
    
    @property
    def collections_map(self):
        return {'voltage': self.get_v_series,
                'current': self.get_i_series,
                'power': self.get_power_series}
    
    def reset_model(self, **kwargs):
        pass

    def init_model(self, **kwargs):
        pass

    def get_params(self):
        pass
    
    def set_params(self, **kwargs):
        pass

    def load_battery_state(self, temp: float, soc: float, soh: float):
        pass

    def build_components(self, components:dict):
        pass

    def compute_generated_heat(self, k:int):
        pass

    def get_results(self, **kwargs):
        """
        Returns a dictionary with results
        """
        results = {}
        k = kwargs['k'] if 'k' in kwargs else None
        var_names = kwargs['var_names'] if 'var_names' in kwargs else None

        for key, func in self.collections_map.items():
            if var_names is not None and key in var_names:
                results[key] = func(k=k)
            
        return results

    def get_v_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        """
        if k is not None:
            assert type(k) == int, \
                "Cannot retrieve load voltage of the electrical model at step K, since it has to be an integer"

            if len(self._v_load_series) > k:
                return self._v_load_series[k]
            else:
                raise IndexError("Load Voltage V of the electrical model at step K not computed yet")
        return self._v_load_series

    def get_i_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        """
        if k is not None:
            assert type(k) == int, \
                "Cannot retrieve load current of the electrical model at step K, since it has to be an integer"

            if len(self._i_load_series) > k:
                return self._i_load_series[k]
            else:
                raise IndexError("Load Current I of the electrical model at step K not computed yet")
        return self._i_load_series

    def get_power_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        """
        if k is not None:
            assert type(k) == int, \
                "Cannot retrieve power of the electrical model at step K, since it has to be an integer"

            if len(self._power_series) > k:
                return self._power_series[k]
            else:
                raise IndexError("Power P of the electrical model at step K not computed yet")
        return self._power_series

    def update_v_load(self, value: float):
        self._v_load_series.append(value)

    def update_i_load(self, value: float):
        self._i_load_series.append(value)

    def update_power(self, value: float):
        self._power_series.append(value)
    
    def clear_collections(self, **kwargs):
        self._v_load_series = [self._v_load_series[-1]]
        self._i_load_series = [self._i_load_series[-1]] 
        self._power_series = [self._power_series[-1]]


class ThermalModel(GenericModel):
    """

    """
    def __init__(self, name: str):
        self._name = name
        self._temp_series = []
        self._heat_series = []
        self._t_amb_series = []

    @property
    def name(self):
        return self._name
    
    @property
    def collections_map(self):
        return {'temperature': self.get_temp_series,
                'heat': self.get_heat_series,
                't_amb': self.get_t_amb_series}

    def reset_model(self, **kwargs):
        pass

    def init_model(self, **kwargs):
        pass

    def load_battery_state(self, **kwargs):
        pass

    def compute_temp(self, **kwargs):
        pass

    def get_results(self, **kwargs):
        """
        Returns a dictionary with all final results
        """
        results = {}
        k = kwargs['k'] if 'k' in kwargs else None
        var_names = kwargs['var_names'] if 'var_names' in kwargs else None
        
        for key, func in self.collections_map.items():
            if var_names is not None and key not in var_names:
                continue
            results[key] = func(k=k)
        
        return results

    def get_temp_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        """
        if k is not None:
            assert type(k) == int, \
                "Cannot retrieve temperature of thermal model at step K, since it has to be an integer"

            if len(self._temp_series) > k:
                return self._temp_series[k]
            else:
                raise IndexError("Temperature of thermal model at step K not computed yet")
        return self._temp_series

    def get_heat_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        """
        if k is not None:
            assert type(k) == int, \
                "Cannot retrieve dissipated heat of thermal model at step K, since it has to be an integer"

            if len(self._heat_series) > k:
                return self._heat_series[k]
            else:
                raise IndexError("Dissipated heat of thermal model at step K not computed yet")
        return self._heat_series
    
    def get_t_amb_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        """
        if k is not None:
            assert type(k) == int, \
                "Cannot retrieve ambient temperature at step K, since it has to be an integer"

            if len(self._t_amb_series) > k:
                return self._t_amb_series[k]
            else:
                raise IndexError("Ambient temperature at step K not computed yet")
        return self._t_amb_series

    def _update_temp(self, value: float):
        self._temp_series.append(value)

    def _update_heat(self, value: float):
        self._heat_series.append(value)
        
    def _update_t_amb(self, value: float):
        self._t_amb_series.append(value)
        
    def update(self, **kwargs):
        """
        Update the thermal model with new values
        """
        self._update_temp(kwargs['temp']) if 'temp' in kwargs else None
        self._update_heat(kwargs['heat']) if 'heat' in kwargs else None
        self._update_t_amb(kwargs['t_amb']) if 't_amb' in kwargs else None
        
    def clear_collections(self, **kwargs):
        """
        Clear data collections of the thermal model
        """
        self._heat_series = [self._heat_series[-1]]
        self._temp_series = [self._temp_series[-1]] 
        self._t_amb_series = [self._t_amb_series[-1]]


class AgingModel(GenericModel):
    """

    """
    def __init__(self, name: str):
        self._name = name
        self._deg_series = []

    @property
    def name(self):
        return self._name
    
    @property
    def collections_map(self):
        return {'degradation': self.get_deg_series}

    def reset_model(self, **kwargs):
        pass

    def init_model(self, **kwargs):
        pass

    def load_battery_state(self, **kwargs):
        pass

    def compute_degradation(self, **kwargs):
        pass

    def get_results(self, **kwargs):
        pass

    def get_deg_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        """
        if k is not None:
            assert type(k) == int, \
                "Cannot retrieve degradation of the aging model at step K, since it has to be an integer"

            if len(self._deg_series) > k:
                return self._deg_series[k]
            else:
                raise IndexError("Degradation of aging model at step K not computed yet")
        return self._deg_series

    def update_deg(self, value: float):
        self._deg_series.append(value)
        
    def clear_collections(self, **kwargs):
        self._deg_series = [self._deg_series[-1]]


