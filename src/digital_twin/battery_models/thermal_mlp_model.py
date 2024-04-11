import pandas as pd

from src.digital_twin.battery_models import ThermalModel
import joblib
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, ConcatDataset


class ThermalMLP(ThermalModel):
    """

    """
    class SimpleDataset(Dataset):
        def __init__(self, x, y):
            super().__init__()
            self.x = x
            self.y = y

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

    # Define the neural network model
    class RegressionModel(nn.Module):

        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )

        def forward(self, x):
            return self.layers(x)

    def __init__(self, components_settings: dict, **kwargs):
        super().__init__(name='MLP_thermal')
        self._settings = components_settings

        if 'ground_temps' in kwargs:
            self._ground_temps = kwargs['ground_temps']
            self._rolling_25 = pd.Series(self._ground_temps).rolling(window=25, min_periods=1).mean()
        else:
            raise AttributeError("The parameter to initialize the attribute of DummyThermal has not benn passed!")

        self._soc = None

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() and self._settings['cuda'] else
            "mps" if torch.backends.mps.is_available() and self._settings['cuda'] else "cpu")

        self._model = self.RegressionModel(input_size=self._settings['input_size'],
                                           hidden_size=self._settings['hidden_size'],
                                           output_size=self._settings['output_size']
                                           ).to(self._device)

        self._model.load_state_dict(torch.load(self._settings['model_state']))
        self._scaler = joblib.load(self._settings['scaler'])

    @property
    def soc(self):
        return self._soc

    def reset_model(self, **kwargs):
        self._temp_series = []

    def init_model(self, **kwargs):
        """
        Initialize the model at timestep t=0 with an initial temperature equal to 25 degC (ambient temperature)
        """
        temp = kwargs['temperature'] if kwargs['temperature'] else 298.15
        heat = 0  # kwargs['dissipated_heat'] if kwargs['dissipated_heat'] else 0

        self.update_temp(temp)
        self.update_heat(heat)

    def load_battery_state(self, **kwargs):
        self._soc = kwargs['soc']

    def compute_temp(self, **kwargs):
        """

        """
        rolling_25 = self._rolling_25[kwargs['k']]
        #rolling_25 = np.mean(self._ground_temps[-25 + kwargs['k'] : kwargs['k']])
        inputs = [kwargs['i'],  kwargs['q'], kwargs['T_amb'], rolling_25]

        inputs = self._scaler.transform(np.array(inputs).reshape(1, -1))
        inputs = torch.tensor(inputs, dtype=torch.float32, device=self._device)

        #print(inputs)

        with torch.no_grad():
            # Generate prediction
            prediction = self._model(inputs).squeeze().tolist()
            #print(prediction)

        return prediction




