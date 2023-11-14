import numpy as np


class VDBSE:
    """

    """
    def __init__(self,
                 capacity: float,
                 soc_time_window: int,
                 moving_step: int,
                 restarts: int,
                 lookup_table,
                 scale_factor: dict,
                 verbose: bool):
        """
        Args:
            capacity ():
            soc_time_window ():
            miving_step ():
            restarts ():
            lookup_table ():
            scale_factor (dict):
            verbose ():
        """
        self._capacity = capacity
        self._soc_time_window = soc_time_window
        self._moving_step = moving_step
        self._restarts = restarts
        self._lookup_table = lookup_table
        self._scale_factors = scale_factor
        self._verbose = verbose

        self._max_soc = 0.
        self._min_soc = 0.
        self._actual_soc = 0.

        # Windows of current and voltage
        self._i_batch = []
        self._v_batch = []

        self._vocv = 0
        self._vocv_series = []
        self._theta = np.zeros(5)
        self._params = {'r0': 0, 'r1': 0, 'c': 0, ...} # TODO da completare

    def _extend_window(self, i, v, dt):
        """
        We are not using the window indices, but we directly store i and v in batches

        Args:
            i (float):
            v (float):
            dt (float):
        """
        self._actual_soc = self._actual_soc - (1 / self._capacity) * i * dt
        self._max_soc = max(self._max_soc, self._actual_soc)
        self._min_soc = min(self._min_soc, self._actual_soc)
        self._i_batch.append(i)
        self._v_batch.append(v)

    def _estimate_vocv(self, dt):
        """

        Args:
            dt ():

        Returns:

        """
        c = self._params['r0'] * self._scale_factors['r0']
        rp = self._params['r1'] * self._scale_factors['r1']
        rs = self._params['c'] * self._scale_factors['c']

        v_ocv_batch = np.zeros(np.shape(self._i_batch))
        # assuming to work with 1D-array
        v_ocv_batch = np.concatenate((np.atleast_1d(self._vocv), vocv_batch))
        v_batch = np.concatenate((np.atleast_1d(last_v), v_batch))
        i_batch = np.concatenate((np.atleast_1d(last_i), i_batch))

        for j in range(len(i_batch)):
            vocv_batch[j] = (((1 / dt) + (1 / (c * rp))) * v_batch[j] - (v_batch[j - 1] / dt) + (vocv_batch[j - 1] / dt)
                             + (((rs / dt) + (1 / c) + (rs / (c * rp))) * i_batch[j]) - (
                                         (rs / dt) * i_batch[j - 1])) / ((1 / dt) + (1 / (c * rp)))

        return vocv_batch[1:]

    def _estimate_v(self, dt):
        """

        Args:
            dt ():

        Returns:

        """
        pass

    def _estimate_params(self, dt):
        """
        Estimation of parameters through a non linear optimation approach.
        Returns: r0, r1, c, soc_tau, q_max
        """
        assert np.size(self._i_batch) == np.size(self._v_batch)

        input_data = np.zeros(len(self._i_batch, 3))
        input_data[:, 0] = self._v_batch
        input_data[:, 1] = self._i_batch
        input_data[:, 2] = -np.cumsum(self._i_batch)

        to_minimize = lambda t: loss_function(t, input_data, dt, self._lookup_table, self._scale_factors)

        # Define optimization options
        if self._verbose:
            display_details = 'iter'
        else:
            display_details = 'off'

        minOptions = {
            'disp': display_details,
            'algorithm': algorithm,
            'gtol': 1e-10,
            'xtol': 1e-10,
            'maxiter': 1000,
            'maxfev': 1000
        }

        # Define bounds and random initialization
        lb = np.array([1e-2, 1e-2, 1e-2, 1e-2, 1e4])
        ub = np.array([1e2, 1e2, 1e2, 1, 1e6])

        ... #TODO
        return [] # or return a dict


    def estimate_soc(self, i, v, dt):
        """
        Core function of the algorithm.

        Args:
            i (float): sample of I at time t
            v (float): sample of V at time t
            dt (float): time delta from previous sample

        Returns: soc estimation

        """
        if (self._max_soc - self._min_soc) < self._soc_time_window: # or < moving_step
            self._extend_window(i=i, v=v, dt=dt)
            return self._actual_soc # estimated with CC

        else:
            self._theta = self._estimate_params(dt=dt)
            self._vocv = self._estimate_vocv(dt)

            self._actual_soc = np.interp(self._vocv_series, lookup_vocv, lookup_soc)

            self._max_soc = 0.
            self._min_soc = 0.
            self._actual_soc = 0.
            self._i_batch = []
            self._v_batch = []

            # TODO: continuare e capire quando sovrascrivere i_batch e v_batch e reinizializzarli per nuova window

            return #something








