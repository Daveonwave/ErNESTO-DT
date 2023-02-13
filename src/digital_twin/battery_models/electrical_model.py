from generic_models import AbstractEquivalentCircuitModel
from src.digital_twin.utils import check_data_unit
from src.digital_twin.units import Unit


class TheveninModel(AbstractEquivalentCircuitModel):
    """
    • 𝑁s𝑚: numero di celle in serie che compongono un singolo modulo;
    • 𝑁𝑝𝑚: numero di celle in parallelo che compongono un singolo modulo;
    • 𝑁s𝑏: numero di moduli in serie che compongono il pacco batteria;
    • 𝑁𝑝𝑏: numero di moduli in parallelo che compongono il pacco batteria;
    • 𝑁s =𝑁s𝑚 x 𝑁 𝑏 : numero di celle totali connesse in serie che compongono il pacco batteria;
    • 𝑁𝑝=𝑁𝑝𝑚 x 𝑁𝑝𝑏 : numero di celle totali connesse in parallelo che compongono il pacco batteria;
    """
    def __init__(self):
        self.ns_cells_module = 0
        self.np_cells_module = 0
        self.ns_modules = 0
        self.np_modules = 0
        self.ns_cells_battery = 0
        self.np_cells_battery = 0

    def get_soc(self):
        pass

    def estimate_r0(self, delta_v0, delta_i):
        """
        Estimation of resistance R0 is conducted with a current pulse test.
        #TODO: added further methods to estimate R0

        :param delta_v0: instantaneous voltage change after the current pulse
        :param delta_i: variation of current in input
        """
        r0 = check_data_unit(delta_v0, Unit.VOLT).magnitude / check_data_unit(delta_i, Unit.AMPERE).magnitude
        return check_data_unit(r0, Unit.OHM)

    def estimate_r1(self, delta_v1, delta_i):
        """
        Estimation of resistance R1 is conducted with a current pulse test.
        #TODO: added further methods to estimate R1

        :param delta_v1: instantaneous voltage change after the current pulse
        :param delta_i: variation of current in input
        """
        r1 = check_data_unit(delta_v1, Unit.VOLT).magnitude / check_data_unit(delta_i, Unit.AMPERE).magnitude
        return check_data_unit(r1, Unit.OHM)

    def estimate_c(self, r1, delta_t):
        """
        Estimation of capacity C1 is conducted with a current pulse test.
        #TODO: added further methods to estimate C1

        :param r1: resistance in parallel with the capacitor
        :param delta_t: time to steady-state which is about 5τ, where τ=R1*C1
        """
        c1 = check_data_unit(delta_t, Unit.SECOND).magnitude / (5 * check_data_unit(r1, Unit.OHM).magnitude)
        return check_data_unit(c1, Unit.FARADAY)