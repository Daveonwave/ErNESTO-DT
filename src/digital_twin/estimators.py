class SOCEstimator:
    """

    """
    def __init__(self, nominal_capacity: float, estimation_mode='CC'):
        """

        Args:
            nominal_capacity ():
            estimation_mode ():
        """
        self._estimation_mode = estimation_mode
        self._q_n = nominal_capacity
        self._soc = 0

    def reset_soc(self, v, v_max, v_min):
        """

        Args:
            v ():
            v_max ():
            v_min ():
        """
        self._soc = (v - v_min) / (v_max - v_min)
        return self._soc

    def compute_soc(self, soc_, i, dt):
        """
        CC = Coulomb Counting
        """
        if self._estimation_mode == "CC":
            self._soc = soc_ + i / (self._q_n * 3600) * dt
            self.crop_soc()
        else:
            raise Exception("Required mode for SoC estimation not existing or not implemented yet.")

        return self._soc

    def crop_soc(self):
        """

        """
        if self._soc < 0:
            self._soc = 0

        if self._soc > 1:
            self._soc = 1
