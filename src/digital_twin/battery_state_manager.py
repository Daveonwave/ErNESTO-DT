class SOCEstimator:
    """

    """
    def __init__(self, nominal_capacity, mode='cc', ):
        self._mode = mode
        self._q_n = nominal_capacity
        self._soc = 0

    def compute_soc(self, soc_, i, dt):
        """

        """
        if self._mode == "cc":
            self._soc = soc_ + i / self._q_n * dt
            self.crop_soc()
            return self._soc
        else:
            raise Exception("Required mode for SoC estimation not existing or not implemented yet.")

    def crop_soc(self):
        """

        """
        if self._soc < 0:
            self._soc = 0

        if self._soc > 1:
            self._soc = 1



class SOHEstimator:
    def __init__(self, mode=None):
        pass