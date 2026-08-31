import numpy as np

from ernesto.adaptation.rls import RLS
from ernesto.adaptation.arx import build_regressor_1RC_at_k
from ernesto.adaptation.ARX_1RC import theta_to_physical_1RC


class ARXRLS1RCEstimator:
    def __init__(
        self,
        Ts=1.0,
        forgetting_factor=0.999,
        P0_scale=1e9,
        bounds=None,
        warmup_samples=20,
    ):
        self.Ts = Ts
        self.bounds = bounds or {}
        self.warmup_samples = warmup_samples

        self.rls = RLS(
            n_params=4,
            theta0=np.zeros(4),
            P0_scale=P0_scale,
            forgetting_factor=forgetting_factor,
        )

        self.voltage_history = []
        self.current_history = []
        self.last_valid_params = None

        self.last_rejection_reason = None
        self.n_updates = 0
        self.n_valid_physical = 0
        self.n_inside_bounds = 0
        self.last_valid_full_params = None

    def reset(self):
        self.rls.reset()
        self.voltage_history = []
        self.current_history = []
        self.last_valid_params = None

    def update(self, voltage, current):
        self.voltage_history.append(float(voltage))
        self.current_history.append(-float(current))

        k = len(self.voltage_history) - 1

        if k < 1:
            return self.last_valid_params


        phi = build_regressor_1RC_at_k(
            self.voltage_history,
            self.current_history,
            k,
        )

        theta, error, gain = self.rls.update(
            phi,
            self.voltage_history[k],
        )
        self.n_updates += 1


        if k < self.warmup_samples:
            return self.last_valid_params
        physical = theta_to_physical_1RC(theta, self.Ts)

        if not physical["valid"]:
            self.last_rejection_reason = physical.get("reason", "invalid_physical")
            return self.last_valid_params

        self.n_valid_physical += 1
        params = {
            "r0": physical["R0"],
            "r1": physical["R1"],
            "c1": physical["C1"],
        }
        self.last_valid_full_params = {
            "r0": physical["R0"],
            "r1": physical["R1"],
            "c1": physical["C1"],
            "ocv": physical["OCV"],
            "alpha": physical["alpha"],
            "tau1": physical["tau1"],
        }

        if not self._inside_bounds(params):
            self.last_rejection_reason = "out_of_bounds"
            return self.last_valid_params

        self.n_inside_bounds += 1
        
        self.last_valid_params = params
        return params

    def estimate_from_batch(self, input_batch):
        for sample in input_batch:
            if "voltage" not in sample or "current" not in sample:
                continue

            self.update(
                voltage=sample["voltage"],
                current=sample["current"],
            )

        if self.last_valid_params is None:
            return None

        return [
            self.last_valid_params["r0"],
            self.last_valid_params["r1"],
            self.last_valid_params["c1"],
        ]

    def _inside_bounds(self, params):
        for name, value in params.items():
            if not np.isfinite(value):
                return False

            if name not in self.bounds:
                continue

            low = self.bounds[name].get("low", -np.inf)
            high = self.bounds[name].get("high", np.inf)

            if value < low or value > high:
                return False

        return True