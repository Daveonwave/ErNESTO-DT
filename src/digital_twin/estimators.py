from src.digital_twin.parameters.vdbse.estimate_soc_vdbse import estimate_soc_vdbse
import numpy as np


class SOCEstimator:
    """

    """
    def __init__(self, i, v, lookup, nominal_capacity, scale_factor, estimation_mode='CC',dt = 0 , moving_step = 0, restarts = 0,
                 soctimewindow = 0, verbose = 0 ):
        self._estimation_mode = estimation_mode
        self._q_n = nominal_capacity
        self._soc = 0
        self._dt = dt
        self._moving_step = moving_step
        self._restarts = restarts
        self._soctimewindow = soctimewindow
        self._verbose = verbose
        self._i = np.zeros(len(i))
        self._i = i
        self._v = np.zeros(len(i))
        self._v = v
        self._lookup = np.zeros(np.shape(lookup))
        self._lookup = lookup
        self._scale_factors = scale_factor

        #self.vdbse = VDBSE(kwargs)



    def compute_soc(self, **kwargs):
        """
        CC = Coulomb Counting
        """
        if self._estimation_mode == "CC":
            self._soc = self._soc + self._i / (self._q_n * 3600) * self._dt
            self.crop_soc()
            return self._soc
        else:
            """
            VDBSE = Voltage Dynamic based SoC Estimation
            """
            if self._estimation_mode == "VDBSE":
               self._soc = estimate_soc_vdbse(self._i, self._v, self._q_n, self._dt,
                                              self._soctimewindow, self._moving_step, self._restarts , self._lookup, self._scale_factors , self._verbose)
            print("vdbse estimstor: ", self._soc)
            return self._soc
             #raise Exception("Required mode for SoC estimation not existing or not implemented yet.")

    def crop_soc(self):
        """

        """
        if self._soc < 0:
            self._soc = 0

        if self._soc > 1:
            self._soc = 1



