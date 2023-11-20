from src.digital_twin.battery_models.vdbse.vdbse import *


class SOCEstimator:
    """

    """
    def __init__(self, nominal_capacity,i , dt, estimation_mode='CC', **kwargs):
        self._estimation_mode = estimation_mode
        self._q_n = nominal_capacity
        self._soc = 0
        self._i = i
        self._dt = dt
        self.vdbse = VDBSE(**kwargs)



    def compute_soc(self, **kwargs):
        """
        CC = Coulomb Counting
        VDBSE = Voltage Dynamic based SoC Estimation
        """
        if self._estimation_mode == "CC":
            self._soc = self._soc + self._i / (self._q_n * 3600) * self._dt
            self.crop_soc()
            return self._soc
        else:
            # TODO: Check if it has meaning, i.e. estimator calling estimate_soc of vdbse
            if self._estimation_mode == "VDBSE":
               self._soc = self.vdbse.estimate_soc(**kwargs)

            return self._soc


    def crop_soc(self):
        """

        """
        if self._soc < 0:
            self._soc = 0

        if self._soc > 1:
            self._soc = 1



