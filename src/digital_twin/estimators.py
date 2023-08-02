class SOCEstimator:
    """

    """
    def __init__(self, nominal_capacity, estimation_mode='CC'):
        self._estimation_mode = estimation_mode
        self._q_n = nominal_capacity
        self._soc = 0

    def compute_soc(self, soc_, i, dt):
        """
        CC = Coulomb Counting
        """
        if self._estimation_mode == "CC":
            self._soc = soc_ + i / (self._q_n * 3600) * dt
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
