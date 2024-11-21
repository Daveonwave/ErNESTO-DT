from src.digital_twin.battery_models.generic_models import AgingModel


class StairStepAging(AgingModel):
    
    def __init__(self,
                 components_settings: dict,
                 init_soc: float = 1.
                 ):
        super().__init__(name="Stair-step aging")
        
        self._aging_step = components_settings['aging_step']
        self._step_length = components_settings['step_length']
        self._k_iters = 0

    def reset_model(self, **kwargs):
        self._k_iters = 0
        self._deg_series = []
    
    def init_model(self, **kwargs):
        self.update_deg(0)
        self._k_iters.append(0)

    def compute_degradation(self, k: int):
        """
        

        Args:
            k (int): k-th iteration of the simulation
        """
        if k % self._step_length == 0:
            deg = self._deg_series[-1] + self._aging_step
            self.update_deg(deg)
            self._k_iters.append(k)
        
        else:
            deg = self._deg_series[-1]
        
        return deg


    def get_results(self, **kwargs):
        k = kwargs['k'] if 'k' in kwargs else None
        return {'degradation': self.get_deg_series(k=k)}
