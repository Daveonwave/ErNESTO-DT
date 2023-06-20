import numpy as np
import rainflow

from src.digital_twin.battery_models.generic_models import AgingModel
from src.digital_twin.battery_models.bolun_components import stress_functions


class BolunModel(AgingModel):
    """
    Bolun model (https://www.researchgate.net/publication/303890624_Modeling_of_Lithium-Ion_Battery_Degradation_for_Cell_Life_Assessment)
    """
    def __init__(self,
                 components_settings: dict,
                 stress_models: dict,
                 units_checker=True
                 ):
        super().__init__(units_checker=units_checker)

        self.alpha_sei = components_settings['SEI']['alpha_sei']
        self.beta_sei = components_settings['SEI']['beta_sei']

        # Collect and build stress factors for both calendar and cyclic aging
        self.calendar_factors = {key: stress_models[key] for key in components_settings['stress_factor']['calendar']}
        self.cyclic_factors = {key: stress_models[key] for key in components_settings['stress_factor']['cyclic']}
        print(self.calendar_factors)
        print(self.cyclic_factors)
        self._compute_calendar_aging()

        #exit()

    def init_model(self, **kwargs):
        pass
    
    def load_battery_state(self, **kwargs):
        pass    
    
    def get_final_results(self, **kwargs):
        pass

    def _compute_combined_degradation(self, f_d, alpha_sei, beta_sei):
        """
        Compute the aging of the battery

        Inputs:
        :param f_d:
        :param alpha_sei:
        :param beta_sei:
        """
        deg = 1 - self.alpha_sei * np.exp(-self.beta_sei * f_d) - (1 - self.alpha_sei) * np.exp(-f_d)
        return deg

    def _compute_calendar_aging(self):
        """

        """
        for factor in self.calendar_factors.keys():
            stress_func = getattr(stress_functions, factor + '_stress')
            print(stress_func)

    def _compute_cyclic_aging(self):
        """

        """
        pass

    def aging_step(self):
        """
        Compute the battery aging due to a single step of simulation with Streamflow method.
        """
        pass

    def aging_period(self, soc_history, temp_history, elapsed_time):
        """
        Compute the battery aging due a longer usage period with Rainflow method.
        """
        pass
