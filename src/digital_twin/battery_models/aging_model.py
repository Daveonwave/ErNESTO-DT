import numpy as np
import rainflow

from src.digital_twin.battery_models.generic_models import AgingModel
from src.digital_twin.battery_models.bolun_components import stress_functions


class BolunModel(AgingModel):
    """
    Bolun model (https://www.researchgate.net/publication/303890624_Modeling_of_Lithium-Ion_Battery_Degradation_for_Cell_Life_Assessment)
    """
    def __init__(self,
                 components_settings: dict,
                 stress_models: dict,
                 ):
        super().__init__()

        self._f_cyc_series = []
        self._f_cal_series = []

        self._alpha_sei = components_settings['SEI']['alpha_sei']
        self._beta_sei = components_settings['SEI']['beta_sei']

        # Collect and build stress factors for both calendar and cyclic aging
        self._calendar_factors = {key: stress_models[key] for key in components_settings['stress_factors']['calendar']}
        self._cyclic_factors = {key: stress_models[key] for key in components_settings['stress_factors']['cyclic']}

        # Stress models constants
        self._stress_models_params = stress_models

        # Fatigue analysis method
        self._cycle_counting_mode = components_settings['cycle_counting_mode']

    def get_f_cal_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        """
        if k is not None:
            assert type(k) == int, \
                "Cannot retrieve calendar aging at step K, since it has to be an integer"

            if len(self._f_cal_series) > k:
                return self._f_cal_series[k]
            else:
                raise IndexError("Calendar aging at step K not computed yet")
        return self._f_cal_series

    def get_f_cyc_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        """
        if k is not None:
            assert type(k) == int, \
                "Cannot retrieve cyclic aging at step K, since it has to be an integer"

            if len(self._f_cyc_series) > k:
                return self._f_cyc_series[k]
            else:
                raise IndexError("Cyclic aging at step K not computed yet")
        return self._f_cyc_series

    def _update_f_cal_series(self, value: float):
        self._f_cal_series.append(value)

    def _update_f_cyc_series(self, value: float):
        self._f_cyc_series.append(value)

    def init_model(self, **kwargs):
        pass
    
    def load_battery_state(self, **kwargs):
        pass    
    
    def get_final_results(self, **kwargs):
        return {}

    def compute_degradation(self, soc_history, temp_history, elapsed_time):
        """
        Compute the aging of the battery

        Inputs:
        :param f_d:
        """
        f_d = 0

        if self._cycle_counting_mode == 'rainflow':
            f_d = self._aging_period(soc_history, temp_history, elapsed_time)

        elif self._cycle_counting_mode == 'streamflow':
            ...
        elif self._cycle_counting_mode == 'fastflow':
            ...
        else:
            raise ValueError("The provided cycle counting method {} is not implemented or not existent"
                             .format(self._cycle_counting_mode))

        deg = 1 - self._alpha_sei * np.exp(-self._beta_sei * f_d) - (1 - self._alpha_sei) * np.exp(-f_d)
        return np.clip(deg, a_min=0., a_max=1.)

    def _compute_calendar_aging(self, curr_time, avg_temp, avg_soc):
        """
        Compute the calendar aging of the battery from the start of the simulation.

        Inputs:
        :param curr_time: wrt the start of the simulation (TODO: what if the battery is not new at the start of the simulation?)
        :param avg_temp:
        :param avg_soc:
        """
        cal_aging = 1

        # For each defined stress model we get the stress function with relative parameters
        for factor in self._calendar_factors.keys():
            stress_func = getattr(stress_functions, factor + '_stress')
            sim_variables = {}

            if factor == 'time':
                sim_variables['t'] = curr_time

            elif factor == 'temperature':
                sim_variables['mean_temp'] = avg_temp

            elif factor == 'soc':
                sim_variables['soc'] = avg_soc

            else:
                raise KeyError("Stress factor {} shouldn't be used for computing calendar aging in Bolun model"
                               .format(factor))

            # Product of all calendar aging factors within the loop
            stress_value = stress_func(**self._stress_models_params[factor], **sim_variables)
            cal_aging = cal_aging * stress_value

        return cal_aging

    def _compute_cyclic_aging(self, cycle_type, cycle_dod, avg_cycle_temp, avg_cycle_soc):
        """
        Compute the cyclic aging of the battery.
        The parameters in input belong to a single cycle identify by means of the Rainflow algorithm.

        Inputs:
        :param cycle_type: value that can be 0.5 or 1.0 and  tells if a cycle is a half or a full cycle
        :param cycle_dod:
        :param avg_cycle_temp:
        :param avg_cycle_soc:
        """
        cyclic_aging = cycle_type

        # For each defined stress model we get the stress function with relative parameters
        for factor in self._cyclic_factors.keys():
            stress_func = getattr(stress_functions, factor + '_stress')
            sim_variables = {}

            if factor == 'dod_bolun':
                sim_variables['dod'] = cycle_dod

            elif factor == 'dod_quadratic':
                sim_variables['dod'] = cycle_dod

            elif factor == 'dod_exponential':
                sim_variables['dod'] = cycle_dod

            elif factor == 'temperature':
                sim_variables['mean_temp'] = avg_cycle_temp

            elif factor == 'soc':
                sim_variables['soc'] = avg_cycle_soc

            else:
                raise KeyError("Stress factor {} shouldn't be used for computing calendar aging in Bolun model"
                               .format(factor))

            # Product of all cyclic aging factors within the loop
            stress_value = stress_func(**self._stress_models_params[factor], **sim_variables)
            cyclic_aging = cyclic_aging * stress_value

        return cyclic_aging

    def _aging_step(self):
        """
        Compute the battery aging due to a single step of simulation with Streamflow method.
        """
        pass

    def _aging_period(self, soc_history, temp_history, elapsed_time):
        """
        Compute the battery aging due a longer usage period with Rainflow algorithm.
        """
        # Compute the calendar aging
        f_cal = self._compute_calendar_aging(curr_time=elapsed_time,
                                             avg_soc=np.mean(soc_history),
                                             avg_temp=np.mean(temp_history))
        self._update_f_cal_series(f_cal)

        # Compute the cyclic aging through rainflow algorithm
        f_cyc = 0
        extracted_cycles = rainflow.extract_cycles(soc_history)

        # With Rainflow we retrieve tuples of (range, mean, count, i_start, i_end) associated to each cycle
        for rng, mean, count, i_start, i_end in extracted_cycles:
            f_cyc += self._compute_cyclic_aging(cycle_type=count,
                                                cycle_dod=rng,
                                                avg_cycle_soc=np.mean(soc_history[i_start:i_end]),
                                                avg_cycle_temp=np.mean(temp_history[i_start:i_end]))
        self._update_f_cyc_series(f_cyc)

        return f_cal + f_cyc

