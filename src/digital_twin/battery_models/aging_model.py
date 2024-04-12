import numpy as np
import rainflow
from enum import Enum

from src.digital_twin.battery_models import AgingModel
from src.digital_twin.battery_models.bolun_components import stress_functions


class BolunModel(AgingModel):
    """
    Bolun model (https://www.researchgate.net/publication/303890624_Modeling_of_Lithium-Ion_Battery_Degradation_for_Cell_Life_Assessment)
    """
    def __init__(self,
                 components_settings: dict,
                 stress_models: dict,
                 init_soc: int = 1
                 ):
        """
        Args:
            components_settings ():
            stress_models ():
            init_soc ():
        """
        super().__init__()

        self._f_cyc_series = []
        self._f_cal_series = []
        self._k_iters = []

        self._alpha_sei = components_settings['SEI']['alpha_sei']
        self._beta_sei = components_settings['SEI']['beta_sei']

        # Collect and build stress factors for both calendar and cyclic aging
        self._calendar_factors = {key: stress_models[key] for key in components_settings['stress_factors']['calendar']}
        self._cyclic_factors = {key: stress_models[key] for key in components_settings['stress_factors']['cyclic']}

        # Stress models constants
        self._stress_models_params = stress_models

        # Fatigue analysis method
        self._cycle_counting_mode = components_settings['cycle_counting_mode']

        if self._cycle_counting_mode == 'streamflow':
            self._streamflow = self.Streamflow(init_soc=init_soc)

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
        self.update_deg(0)
        self._update_f_cyc_series(0)
        self._update_f_cal_series(0)
        self._k_iters.append(0)

    def compute_degradation(self, soc_history, temp_history, elapsed_time, k):
        """
        Compute the aging of the battery

        Inputs:
        :param f_d:
        """
        f_d = 0

        if self._cycle_counting_mode == 'rainflow':
            f_d = self._aging_period(soc_history, temp_history, elapsed_time)

        elif self._cycle_counting_mode == 'streamflow':
            f_d = self._aging_step(soc_history=soc_history, temp_history=temp_history, t=elapsed_time)

        elif self._cycle_counting_mode == 'fastflow':
            ...

        else:
            raise ValueError("The provided cycle counting method {} is not implemented or not existent"
                             .format(self._cycle_counting_mode))

        # Compute degradation considering the SEI film factor
        deg = np.clip(1 - self._alpha_sei * np.exp(-self._beta_sei * f_d) - (1 - self._alpha_sei) * np.exp(-f_d),
                      a_min=0., a_max=1.)
        self.update_deg(deg)
        self._k_iters.append(k)
        return deg

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

    def _aging_step(self, soc_history, temp_history, t):
        """
        Compute the battery aging due to a single step of simulation with Streamflow method.

        Args:
            soc ():
            temp ():
            t ():

        Returns: the sum of both cyclic and calendar aging  for the given period
        """
        f_cal = self._compute_calendar_aging(curr_time=t,
                                             avg_soc=np.mean(soc_history),
                                             avg_temp=np.mean(temp_history))
        self._update_f_cal_series(f_cal)

        is_charging = soc_history[-1] > soc_history[-2]
        if is_charging:
            expected_end = 0.5 * (1 - soc_history[-1])
        else:
            expected_end = 0.5 * soc_history[-1]

        if len(soc_history) % self._streamflow._reset_every == 0:
            self._streamflow = self.Streamflow(init_soc=soc_history[-1])

        soc_means, ranges, n_samples_arr, is_valid, _, start_indexes, temp_means, to_invalid = \
            self._streamflow.step(actual_value=soc_history[-1],
                                  expected_end=expected_end,
                                  second_signal_value=temp_history[-1])

        ranges[ranges == 0] += 1e-6
        cycle_types = 0.5
        valid_n_cycles = len(soc_means[is_valid])

        valid_temp_means = temp_means[is_valid]
        valid_f_cyc_arr = self._compute_cyclic_aging(cycle_type=cycle_types,
                                                     cycle_dod=ranges[is_valid],
                                                     avg_cycle_soc=soc_means[is_valid],
                                                     avg_cycle_temp=valid_temp_means)
        f_cyc = np.sum(valid_f_cyc_arr)

        self._update_f_cyc_series(f_cyc)

        if len(to_invalid) != 0:
            invalid_soc_means = soc_means[to_invalid]
            invalid_ranges = ranges[to_invalid]
            invalid_sample_arr = n_samples_arr[to_invalid]
            invalid_start_indexes = start_indexes[to_invalid]
            invalid_n_cycles = len(invalid_soc_means)
            invalid_temp_means = temp_means[to_invalid]
            f_cyc_invalid = np.sum(
                self._compute_cyclic_aging(cycle_type=cycle_types,
                                           cycle_dod=invalid_ranges,
                                           avg_cycle_soc=invalid_soc_means,
                                           avg_cycle_temp=invalid_temp_means))
            self._streamflow._stream_f_cyc_past += f_cyc_invalid

        return f_cal + f_cyc + self._streamflow._stream_f_cyc_past

    def _aging_period(self, soc_history, temp_history, elapsed_time):
        """
        Compute the battery aging due a longer usage period with Rainflow algorithm.

        Args:
            soc_history (list):
            temp_history (list):
            elapsed_time (list):

        Returns: the sum of both cyclic and calendar aging  for the given period
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

    def get_final_results(self, **kwargs):
        """
        Returns a dictionary with all final results
        TODO: selection of results by label from config file?
        """
        return {'iteration': self._k_iters,
                'cyclic_aging': self._f_cyc_series,
                'calendar_aging': self._f_cal_series,
                'degradation': self.get_deg_series()
                }

    class Streamflow:
        """
        Implementation of our cycle counting algorithm, that is able to perform in an online manner without considering
        every new sample the whole soc and temperature history. It's inspired to the rainflow cycle counting algorithm.
        """
        class Direction(Enum):
            UP = 1
            DOWN = 2

        def __init__(self,
                     init_soc=0,
                     subsample=False,
                     interpolate='linear',
                     expected_cycle_num=500,
                     cycle_num_increment=500):
            """
            Args:
                init_soc (int):
                subsample (bool):
                interpolate (str):
                expected_cycle_num (int): number of fixed initial cycles considered
                cycle_num_increment (int): amount of cycles used to increment the size of initial arrays
            """
            # K-th cycle of the simulation
            self._cycle_k = -1

            # Direction can be 1 -> up, 2 -> down
            self._directions = np.zeros(expected_cycle_num, dtype=self.Direction)

            # Signals mean values
            self._mean_values = np.zeros(expected_cycle_num)
            self._second_signal_means = np.zeros(expected_cycle_num)

            # Inferior and superior of cycles and range considered as (max-min)
            self._min_max_vals = np.zeros((expected_cycle_num, 2))
            self._range_size = np.zeros(expected_cycle_num)

            # number of samples in the cycle
            self._number_of_samples = np.zeros(expected_cycle_num, dtype=int)
            self._start_cycles = np.zeros(expected_cycle_num, dtype=int)
            self._end_cycles = np.zeros(expected_cycle_num, dtype=int)
            # TODO: find the right value
            self._reset_every = 50000

            # Cycles that are waiting to be completed
            self._is_valid = np.zeros(expected_cycle_num, dtype=bool)
            # Mask to hide cycles allocated for efficiency purpose only
            self._is_used = np.zeros(expected_cycle_num, dtype=bool)

            self._aux_enum = np.linspace(0, expected_cycle_num - 1, expected_cycle_num, dtype=int)
            self._last_value = init_soc

            # To manage edge cases
            self._subsample = subsample
            # Only linear interpolation for now
            self._interpolate = interpolate
            # For performance purpose only
            self._cycle_num_increment = cycle_num_increment

            self._is_init = True
            self._iteration = 0
            self._stream_f_cyc_past = 0

        def step(self,
                 actual_value,
                 expected_end,
                 second_signal_value=None,
                 return_valid_only=True,
                 return_unvalidated_list=True):
            """
            Args:
                actual_value ():
                expected_end ():
                second_signal_value ():
                return_valid_only ():
                return_unvalidated_list ():
            """
            change_direction = False
            # Obtain information about closed cycles
            to_invalid = []
            if (self._is_init or
                    (self._directions[self._cycle_k] == 1 and actual_value < self._last_value) or
                    (self._directions[self._cycle_k] == 2 and actual_value > self._last_value)):
                change_direction = True
                self._is_init = False

            # Case in which there is a change of current direction wrt actual cycle -> creation of new cycle
            if change_direction:
                self._create_new_cycle(value=actual_value, second_signal_value=second_signal_value)

            # Direction of the cycle doesn't change
            else:
                self._update_existent_cycle(value=actual_value, expected_end=expected_end,
                                            second_signal_value=second_signal_value)

            valid_and_used = np.logical_and(self._is_used, self._is_valid)
            # self.number_of_samples[valid_and_used] += 1  # TODO this could be a problem
            self._last_value = actual_value
            self._iteration += 1

            return (self._mean_values[self._is_used],
                    self._range_size[self._is_used],
                    self._number_of_samples[self._is_used],
                    self._is_valid[self._is_used],
                    change_direction,
                    self._start_cycles[self._is_used],
                    self._second_signal_means[self._is_used],
                    to_invalid)

        def _create_new_cycle(self, value, second_signal_value=None):
            """

            Args:
                value ():
                second_signal_value ():
            """
            self._cycle_k = np.sum(self._is_used)

            if self._cycle_k >= self._range_size.shape[0]:
                self._expand()

            # Enable the new cycle and set the direction
            self._is_valid[self._cycle_k] = True
            self._is_used[self._cycle_k] = True

            if value < self._last_value:
                self._directions[self._cycle_k] = 2
            else:
                self._directions[self._cycle_k] = 1

            # Set the min and max values of the new cycle
            min_val, max_val = min(value, self._last_value), max(value, self._last_value)
            self._min_max_vals[self._cycle_k] = (min_val, max_val)
            self._range_size[self._cycle_k] = max_val - min_val

            self._mean_values[self._cycle_k] = value
            if second_signal_value is not None:
                self._second_signal_means[self._cycle_k] = second_signal_value
            self._number_of_samples[self._cycle_k] = 1

            self._start_cycles[self._cycle_k] = self._iteration

            # self.mean_values[self.actual_cycle] = 0
            # self.number_of_samples[self.actual_cycle] = 0

        def _update_existent_cycle(self, value: float, expected_end: float, second_signal_value=None):
            """

            Args:
                value ():
                expected_end ():
                second_signal_value ():
            """
            # Get indices of cycles used, valid and with the same direction of current cycle
            valid_used_correct_direction = (self._is_used and self._is_valid and
                                            (self._directions == self._directions[self._cycle_k]))

            is_direction_up = self._directions[self._cycle_k] == 1
            min_max_index = 1 if is_direction_up else 0
            arg_function = np.argmax if is_direction_up else np.argmin

            indices = self._get_indices_by_direction(direction=self._directions[self._cycle_k],
                                                     value=value,
                                                     indices_range=valid_used_correct_direction)

            expected_end_indices = self._get_indices_by_direction(direction=self._directions[self._cycle_k],
                                                                  value=expected_end,
                                                                  indices_range=valid_used_correct_direction)

            """
            up_indexes = (self._min_max_vals[valid_used_correct_direction][:, 1] > self._last_value
                          ) & (self._min_max_vals[valid_used_correct_direction][:, 1] < actual_value)

            down_indexes = (self._min_max_vals[valid_used_correct_direction][:, 0] < self._last_value
                            ) & (self._min_max_vals[valid_used_correct_direction][:, 0] > actual_value)
            indexes = up_indexes if is_direction_up == 1 else down_indexes

            expected_indexes_up = (self._min_max_vals[valid_used_correct_direction][:, 1] > actual_value
                                   ) & (self._min_max_vals[valid_used_correct_direction][:, 1] < expected_end)
            expected_indexes_down = (self._min_max_vals[valid_used_correct_direction][:, 0] < actual_value
                                     ) & (self._min_max_vals[valid_used_correct_direction][:, 0] > expected_end)
            expected_end_indexes = expected_indexes_up if is_direction_up else expected_indexes_down
            """
            if indices.any():
                aux = self._aux_enum[valid_used_correct_direction]

                # If something will fall later, invalid all the current falling cycles
                if expected_end_indices.any():
                    to_invalid = aux[indices]
                    self._is_valid[to_invalid] = False

                # Nothing is falling, so find the biggest that is falling
                else:
                    # Indices is boolean, therefore aux[indices] filter
                    self._is_valid[self._cycle_k] = False
                    self._cycle_k = arg_function(self._min_max_vals[aux[indices]][:, min_max_index])
                    to_invalid = np.concatenate((aux[indices][0:self._cycle_k],
                                                 aux[indices][self._cycle_k + 1:len(aux[indices])]))
                    self._is_valid[to_invalid] = False
                    self._is_valid[self._cycle_k] = True

            # Update of the mean and the ranges
            self._mean_values[self._cycle_k] = ((self._mean_values[self._cycle_k] *
                                                self._number_of_samples[self._cycle_k] + value) /
                                                (self._number_of_samples[self._cycle_k] + 1))
            if second_signal_value is not None:
                self._second_signal_means[self._cycle_k] = \
                    (self._second_signal_means[self._cycle_k] * self._number_of_samples[self._cycle_k] +
                     second_signal_value) / (self._number_of_samples[self._cycle_k] + 1)

            # Update min_max values considering the new value and the range
            self._min_max_vals[self._cycle_k] = (min(value, self._min_max_vals[self._cycle_k][0]),
                                                 max(value, self._min_max_vals[self._cycle_k][1]))
            self._range_size[self._cycle_k] = (self._min_max_vals[self._cycle_k][1] -
                                               self._min_max_vals[self._cycle_k][0])

            # bounds_tuple = self._min_max_vals[self._cycle_k]
            # min_val, max_val = min(bounds_tuple[0], actual_value), max(bounds_tuple[1], actual_value)
            # self._min_max_vals[self._cycle_k] = (min_val, max_val)

            # TODO potrebbe risolvere non monotonia
            self._number_of_samples[self._cycle_k] += 1

        def _get_indices_by_direction(self, direction: Direction, value: float, indices_range: np.array = None):
            """

            Args:
                direction (Direction):
                value ():
                indices_range ():

            Returns: indices of desired subset of cycles
            """
            if direction == 1:
                indices = np.logical_and(self._min_max_vals[indices_range][:, 1] > self._last_value,
                                         self._min_max_vals[indices_range][:, 1] < value)
            elif direction == 2:
                indices = np.logical_and(self._min_max_vals[indices_range][:, 0] < self._last_value,
                                         self._min_max_vals[indices_range][:, 0] > value)
            else:
                raise ValueError("The specified direction does not exist!")

            return indices

        def _expand(self):
            """

            """
            # more cycles than the one pre-allocated, need to allocate new ones
            self._mean_values = np.concatenate(
                (self._mean_values, np.zeros(self._cycle_num_increment)))
            self.second_signal_means = np.concatenate(
                (self._second_signal_means, np.zeros(self._cycle_num_increment)))
            self.range_size = np.concatenate(
                (self._range_size, np.zeros(self._cycle_num_increment)))
            self.min_max_vals = np.vstack(
                (self._min_max_vals, np.zeros((self._cycle_num_increment, 2))))
            self._number_of_samples = np.concatenate(
                (self._number_of_samples, np.zeros(self._cycle_num_increment, dtype=int)))
            self.start_cycles = np.concatenate(
                (self._start_cycles, np.zeros(self._cycle_num_increment, dtype=int)))
            self.end_cycles = np.concatenate(
                (self._end_cycles, np.zeros(self._cycle_num_increment, dtype=int)))
            self._directions = np.concatenate(
                (self._directions, np.zeros(self._cycle_num_increment, dtype=int)))
            self._is_valid = np.concatenate(
                (self._is_valid, np.zeros(self._cycle_num_increment, dtype=bool)))
            self._is_used = np.concatenate(
                (self._is_used, np.zeros(self._cycle_num_increment, dtype=bool)))
            self._aux_enum = np.concatenate(
                (self._aux_enum, np.linspace(self._aux_enum.shape[0], self._aux_enum.shape[0] +
                                             self._cycle_num_increment - 1,
                                             self._cycle_num_increment, dtype=int)))
