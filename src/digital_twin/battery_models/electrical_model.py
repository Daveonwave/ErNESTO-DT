from generic_models import AbstractEquivalentCircuitModel
from src.digital_twin.utils import check_data_unit
from src.digital_twin.units import Unit
from src.digital_twin.battery_models.ecm_components.resistor import Resistor
from src.digital_twin.battery_models.ecm_components.rc_parallel import ResistorCapacitorParallel
from src.digital_twin.battery_models.ecm_components.ocv_generator import OCVGenerator


class TheveninModel(AbstractEquivalentCircuitModel):
    """
    CLass
    """
    R0 = 10
    R1 = 5
    C = 100
    def __init__(self, **params):
        """
        • 𝑁s𝑚: numero di celle in serie che compongono un singolo modulo;
        • 𝑁𝑝𝑚: numero di celle in parallelo che compongono un singolo modulo;
        • 𝑁s𝑏: numero di moduli in serie che compongono il pacco batteria;
        • 𝑁𝑝𝑏: numero di moduli in parallelo che compongono il pacco batteria;
        • 𝑁s =𝑁s𝑚 x 𝑁 𝑏 : numero di celle totali connesse in serie che compongono il pacco batteria;
        • 𝑁𝑝=𝑁𝑝𝑚 x 𝑁𝑝𝑏 : numero di celle totali connesse in parallelo che compongono il pacco batteria;
        """
        self.ns_cells_module = 0
        self.np_cells_module = 0
        self.ns_modules = 0
        self.np_modules = 0
        self.ns_cells_battery = 0
        self.np_cells_battery = 0

        self._v_load_series = []
        self._i_load_series = []
        self._times = []

        # Components of the Thevenin equivalent circuit
        self.r0 = Resistor(name='R0', resistance=self.R0)
        self.rc = ResistorCapacitorParallel(name='RC', capacity=self.C, resistance=self.R1)
        self.ocv_gen = OCVGenerator(name='OCV')

    @property
    def v_load_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        """
        if k:
            assert type(k) == int, \
                "Cannot retrieve voltage of {} at step K, since it has to be an integer".format(self._name)

            if len(self._v_load_series) > k:
                return self._v_load_series[k]
            else:
                raise IndexError("Voltage V of {} at step K not computed yet".format(self._name))
        return self._v_load_series

    @property
    def i_load_series(self, k=None):
        """
        Getter of the specific value at step K, if specified, otherwise of the entire collection
        """
        if k:
            assert type(k) == int, \
                "Cannot retrieve current of {} at step K, since it has to be an integer".format(self._name)

            if len(self._i_load_series) > k:
                return self._i_load_series[k]
            else:
                raise IndexError("Current I of {} at step K not computed yet".format(self._name))
        return self._i_load_series

    def update_v_load(self, value:float):
        self._v_load_series.append(value)

    def update_i_load(self, value:float):
        self._v_load_series.append(value)

    def load_battery_state(self, soc, soh):
        """
        Update the SoC and SoH for the current simulation step
        #TODO: update intrinsic status of circuital components (i.e. decay of resistor)
        """
        self.ocv_gen.soc(value=soc)
        # self.r0.soc(value=soc) -> I'll probably need to do this one day
        # self.rc.soc(value=soc)

    def solve_components_cv_mode(self, v_load, dt, k):
        """
        CV mode
        #TODO: load has to be defined better
        """
        # Solve the equation to get I
        r0 = self.r0.resistance.magnitude
        r1 = self.rc.resistance.magnitude
        c = self.rc.capacity.magnitude

        eq_factor = (dt * c * r1) / (r0 * c * r1 + dt * (r1 + r0))
        term_1 = - (1/dt + 1/(c * r1)) * v_load
        term_2 = 1/dt * self._v_load_series[k-1]
        term_3 = (1/dt + 1/(c * r1)) * self.ocv_gen.v_series(k)
        term_4 = - 1/dt * self.ocv_gen.v_series(k-1)
        term_5 = r0 / dt * self._i_load_series[k-1]
        i = eq_factor * (term_1 + term_2 + term_3 + term_4 + term_5)

        # Compute V_r0
        v_r0 = self.r0.compute_v(i=i)

        # Compute V_c
        v_rc = self.rc.compute_v(v_ocv=self.ocv_gen.v_series(k), v_r0=v_r0, v=v_load)

        # Compute I_r1 and I_c for the RC parallel
        i_r1 = self.rc.compute_i_r1(v_rc=v_rc)
        i_c = self.rc.compute_i_c(i=i, i_r1=i_r1)

        # Update the collections of variables of ECM components
        self.rc.update_step_variables(v=v_rc, i_r1=i_r1, i_c=i_c, dt=dt, k=k)
        self.r0.update_step_variables(v=v_r0, dt=dt, k=k)
        self.update_i_load(value=i)

        return i

    def solve_components_cc_mode(self, i_load, dt, k):
        """
        CC mode
        #TODO: load has to be defined better
        """
        # Compute V_r0
        v_r0 = self.r0.compute_v(i=i_load)

        # Solve the equation to get V
        r0 = self.r0.resistance.magnitude
        r1 = self.rc.resistance.magnitude
        c = self.rc.capacity.magnitude

        eq_factor = dt * c * r1 / (dt + c * r1)
        term_1 = 1/dt * self._v_load_series[k-1]
        term_2 = (1/dt + 1/(c * r1)) * self.ocv_gen.get_v(k)
        term_3 = -1/dt * self.ocv_gen.get_v(k-1)
        term_4 = - (r0 / (c * r1) + r0 / dt + 1 / c) * i_load
        term_5 = r0 / dt * self._i_load_series[k-1]
        v = eq_factor * (term_1 + term_2 + term_3 + term_4 + term_5)

        # Compute V_rc
        v_rc = self.rc.compute_v(v_ocv=self.ocv_gen.get_v(k), v_r0=v_r0, v=v)

        # Compute I_r1 and I_c for the RC parallel
        i_r1 = self.rc.compute_i_r1(v_rc=v_rc)
        i_c = self.rc.compute_i_c(i=i_load, i_r1=i_r1)

        # Update the collections of variables of ECM components
        self.rc.update_step_variables(v=v_rc, i_r1=i_r1, i_c=i_c, dt=dt, k=k)
        self.r0.update_step_variables(v=v_r0, dt=dt, k=k)
        self.update_v_load(value=v)

        return v

    def get_soc(self):
        pass

    def estimate_r0(self, delta_v0, delta_i):
        """
        Estimation of resistance R0 is conducted with a current pulse test.
        #TODO: add further methods to estimate R0

        :param delta_v0: instantaneous voltage change after the current pulse
        :param delta_i: variation of current in input
        """
        r0 = check_data_unit(delta_v0, Unit.VOLT).magnitude / check_data_unit(delta_i, Unit.AMPERE).magnitude
        return check_data_unit(r0, Unit.OHM)

    def estimate_r1(self, delta_v1, delta_i):
        """
        Estimation of resistance R1 is conducted with a current pulse test.
        #TODO: add further methods to estimate R1

        :param delta_v1: instantaneous voltage change after the current pulse
        :param delta_i: variation of current in input
        """
        r1 = check_data_unit(delta_v1, Unit.VOLT).magnitude / check_data_unit(delta_i, Unit.AMPERE).magnitude
        return check_data_unit(r1, Unit.OHM)

    def estimate_c(self, r1, delta_t):
        """
        Estimation of capacity C1 is conducted with a current pulse test.
        #TODO: add further methods to estimate C1

        :param r1: resistance in parallel with the capacitor
        :param delta_t: time to steady-state which is about 5τ, where τ=R1*C1
        """
        c1 = check_data_unit(delta_t, Unit.SECOND).magnitude / (5 * check_data_unit(r1, Unit.OHM).magnitude)
        return check_data_unit(c1, Unit.FARADAY)