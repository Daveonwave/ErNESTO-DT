from src.digital_twin.battery_models.generic_models import ElectricalModel
from src.digital_twin.parameters.data_checker import craft_data_unit
from src.digital_twin.parameters.units import Unit
from src.digital_twin.battery_models.ecm_components.resistor import Resistor
from src.digital_twin.battery_models.ecm_components.rc_parallel import ResistorCapacitorParallel
from src.digital_twin.battery_models.ecm_components.ocv_generator import OCVGenerator
from src.digital_twin.parameters.variables import instantiate_variables


class TheveninModel(ElectricalModel):
    """
    CLass
    """
    def __init__(self,
                 components_settings: dict,
                 sign_convention='active'
                 ):
        """
        • 𝑁s𝑚: numero di celle in serie che compongono un singolo modulo;
        • 𝑁𝑝𝑚: numero di celle in parallelo che compongono un singolo modulo;
        • 𝑁s𝑏: numero di moduli in serie che compongono il pacco batteria;
        • 𝑁𝑝𝑏: numero di moduli in parallelo che compongono il pacco batteria;
        • 𝑁s =𝑁s𝑚 x 𝑁 𝑏 : numero di celle totali connesse in serie che compongono il pacco batteria;
        • 𝑁𝑝=𝑁𝑝𝑚 x 𝑁𝑝𝑏 : numero di celle totali connesse in parallelo che compongono il pacco batteria;
        """
        super().__init__()
        self.sign_convention = sign_convention

        self.ns_cells_module = 0
        self.np_cells_module = 0
        self.ns_modules = 0
        self.np_modules = 0
        self.ns_cells_battery = 0
        self.np_cells_battery = 0

        [r0, r1, c, v_ocv] = instantiate_variables(components_settings)

        self.r0 = Resistor(name='R0', resistance=r0)
        self.rc = ResistorCapacitorParallel(name='RC', resistance=r1, capacity=c)
        self.ocv_gen = OCVGenerator(name='OCV', ocv_potential=v_ocv)

    def reset_model(self):
        self._v_load_series = []
        self._i_load_series = []
        # self._times = []

    def init_model(self, **kwargs):
        """
        Initialize the model at t=0
        """
        v = kwargs['voltage'] if kwargs['voltage'] else 0
        i = kwargs['current'] if kwargs['current'] else 0
        p = v * i

        self.update_v_load(v)
        self.update_i_load(i)
        self.update_power(p)
        # self.update_times(0)

        self.r0.init_component()
        self.rc.init_component()
        self.ocv_gen.init_component()

    def load_battery_state(self, temp=None, soc=None, soh=None):
        """
        Update the SoC and SoH for the current simulation step
        """
        for component in [self.r0, self.rc, self.ocv_gen]:
            if temp is not None:
                component.temp = temp
            if soc is not None:
                component.soc = soc
            if soh is not None:
                component.soh = soh

        # self.r0.soc(value=soc) -> I'll probably need to do this one day
        # self.rc.soc(value=soc)

    def step_voltage_driven(self, v_load, dt, k):
        """
        CV mode
        """
        # Solve the equation to get I
        r0 = self.r0.resistance
        r1 = self.rc.resistance
        c = self.rc.capacity
        v_ocv = self.ocv_gen.ocv_potential
        v_ocv_ = self.ocv_gen.get_v_series(k=-1)

        eq_factor = (dt * c * r1) / (r0 * c * r1 + dt * (r1 + r0))
        term_1 = - (1/dt + 1/(c * r1)) * v_load
        term_2 = 1/dt * self.get_v_load_series(k=-1)
        term_3 = (1/dt + 1/(c * r1)) * v_ocv
        term_4 = - 1/dt * v_ocv_
        term_5 = r0 / dt * self.get_i_load_series(k=-1)
        i = eq_factor * (term_1 + term_2 + term_3 + term_4 + term_5)

        # Compute V_r0
        v_r0 = self.r0.compute_v(i=i)

        # Compute V_c
        v_rc = self.rc.compute_v(v_ocv=v_ocv, v_r0=v_r0, v=v_load)

        # Compute I_r1 and I_c for the RC parallel
        i_r1 = self.rc.compute_i_r1(v_rc=v_rc)
        i_c = self.rc.compute_i_c(i=i, i_r1=i_r1)

        # Compute power
        power = v_load * i
        if self.sign_convention == 'passive':
            power = -power

        # Update the collections of variables of ECM components
        self.r0.update_step_variables(r0=r0, v_r0=v_r0, dt=dt, k=k)
        self.rc.update_step_variables(r1=r1, c=c, v_rc=v_rc, i_r1=i_r1, i_c=i_c, dt=dt, k=k)
        self.ocv_gen.update_v(value=v_ocv)
        self.update_i_load(value=i)
        self.update_v_load(value=v_load)
        self.update_power(value=power)

        return i

    def step_current_driven(self, i_load, dt, k):
        """
        CC mode
        """
        if self.sign_convention == 'passive':
            i_load = -i_load

        # Compute V_r0
        v_r0 = self.r0.compute_v(i=i_load)

        # Solve the equation to get V
        r0 = self.r0.resistance
        r1 = self.rc.resistance
        c = self.rc.capacity
        v_ocv = self.ocv_gen.ocv_potential
        v_ocv_ = self.ocv_gen.get_v_series(k=-1)

        #print('r0: ', self.r0.resistance)
        #print('r1: ', self.rc.resistance)
        #print('c1: ', self.rc.capacity)
        #print('ocv: ', self.ocv_gen.ocv_potential)

        """
        eq_factor = 1 / (c * r1 - dt)
        term_1 = c * r1 * self.get_v_load_series(k=-1)
        term_2 = (c * r1 - dt) * v_ocv
        term_3 = - c * r1 * v_ocv_
        term_4 = (dt * r1 + dt * r0 + c * r0 * r1) * i_load
        term_5 = - c * r0 * r1 * self.get_i_load_series(k=-1)
        v = eq_factor * (term_1 + term_2 + term_3 + term_4 + term_5)

        """
        eq_factor = dt * c * r1 / (dt + c * r1)
        term_1 = 1/dt * self.get_v_load_series(k=-1)
        term_2 = (1/dt + 1/(c * r1)) * v_ocv
        term_3 = -1/dt * v_ocv_
        term_4 = - (r0 / (c * r1) + r0 / dt + 1 / c) * i_load
        term_5 = r0 / dt * self.get_i_load_series(k=-1)
        v = eq_factor * (term_1 + term_2 + term_3 + term_4 + term_5)

        # Compute V_rc
        v_rc = self.rc.compute_v(v_ocv=v_ocv, v_r0=v_r0, v=v)

        # Compute I_r1 and I_c for the RC parallel
        i_r1 = self.rc.compute_i_r1(v_rc=v_rc)
        i_c = self.rc.compute_i_c(i=i_load, i_r1=i_r1)

        # Compute power
        power = v * i_load
        if self.sign_convention == 'passive':
            power = -power

        # Update the collections of variables of ECM components
        self.r0.update_step_variables(r0=r0, v_r0=v_r0, dt=dt, k=k)
        self.rc.update_step_variables(r1=r1, c=c, v_rc=v_rc, i_r1=i_r1, i_c=i_c, dt=dt, k=k)
        self.ocv_gen.update_v(value=v_ocv)
        self.update_v_load(value=v)
        self.update_i_load(value=i_load)
        self.update_power(value=power)
        return v

    def step_power_driven(self, p_load):
        """

        """
        raise NotImplementedError()

    def compute_generated_heat(self, k=-1):
        """
        Compute the generated heat that can be used to feed the thermal model (when required).
        For Thevenin first order circuit it is: [P = V * I + V_rc * I_r1].

        Inputs:
        :param k: step for which compute the heat generation
        """
        # return self.r0.get_r0_series(k=k) * self.get_i_load_series(k=k)**2 + \
        #           self.rc.get_r1_series(k=k) * self.rc.get_i_r1_series(k=k)**2
        return self.r0.get_r0_series(k=k) * self.get_i_load_series(k=k) ** 2

    def get_final_results(self, **kwargs):
        """
        Returns a dictionary with all final results
        TODO: selection of results by label from config file?
        """
        return {'Voltage [V]': self._v_load_series,
                'Current [A]': self._i_load_series,
                'Power [W]': self._power_series,
                'Vocv': self.ocv_gen.get_v_series(),
                'R0': self.r0.get_r0_series(),
                'R1': self.rc.get_r1_series(),
                'C': self.rc.get_c_series()}
