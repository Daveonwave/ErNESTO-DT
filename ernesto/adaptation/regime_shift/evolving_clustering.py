import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import mahalanobis
from typing import Dict, List, Optional, Tuple, Any
import warnings

from ernesto.adaptation.base_adapter import BaseAdapter
from ernesto.adaptation.optimizer import Optimizer
from ernesto.adaptation.regime_shift.parameter_space import ParameterSpace
from ernesto.postprocessing.interactive_plot import ParameterSpaceVisualizer
from ernesto.adaptation.regime_shift.stats import *
from ernesto.adaptation.arx_rls_estimator import ARXRLS1RCEstimator


class EvolvingClusteringRoutine(BaseAdapter):
    """
    Adaptation routine for regime shifts in the parameter space.
    """
    def __init__(self, 
                 model_config: dict,
                 sim_config: dict,
                 clusters_folder: str,
                 batch_size: int = None,
                 enable_adaptation: bool = True,
                 output_folder: str = None,
                 render: bool = False,
                 **kwargs
                 ):
        """
        Initialize the adaptation routine for regime shifts.

        Args:
            model_config (dict): Configuration for the model.
            sim_config (dict): Configuration for the simulation.
            clusters_folder (str): Path to the folder containing cluster data.
            batch_size (int, optional): _description_. Defaults to None.
            enable_adaptation (bool, optional): _description_. Defaults to True.
            immediate_adaptation (bool, optional): _description_. Defaults to False.    
            output_folder (str, optional): _description_. Defaults to None.
            render (bool, optional): _description_. Defaults to False.
        """
        super().__init__(**kwargs)
        self._enable_adaptation = enable_adaptation
        self._n_faulty_clusters = 1
        self._adaptation_options = sim_config.get('adaptation', {})
        
        # Simulation variables
        self._input_batch = []
        self._batch_size = batch_size if batch_size is not None else sim_config['optimizer']['batch_size']
        self._init_state = {}
        
        # Optimizer parameters
        self._optimizer = Optimizer(battery_config={'models_config': model_config,
                                                    'battery_options': sim_config['battery']},
                                    **sim_config['optimizer'],
                                    enabled_adaptation=self._enable_adaptation,
                                    )
        
        self._param_space = ParameterSpace(parameter_space_config=sim_config['parameter_space'],
                                           clusters_folder=clusters_folder,
                                           output_folder=output_folder)
        
        # Add time tracking for visualization
        self._adaptation_time_step = 0
        self._data_history = dict(zip(self.get_domain_vars() + self._param_space.param_variables + ['cluster', 'is_outlier', 'time', 'active_cluster'],
                                      [list() for _ in range(len(self.get_domain_vars() + self._param_space.param_variables + ['cluster', 'is_outlier', 'time', 'active_cluster']))]))

        self._estimation_method = sim_config.get(
            "estimation_method",
            "optimizer",
        )

        if self._estimation_method == "arx_rls":
            arx_config = sim_config.get("arx_rls", {})

            self._arx_rls = ARXRLS1RCEstimator(
                Ts=arx_config.get("Ts", 1.0),
                forgetting_factor=arx_config.get("forgetting_factor", 0.999),
                P0_scale=arx_config.get("P0_scale", 1e9),
                warmup_samples=arx_config.get("warmup_samples", 20),
                bounds=sim_config["optimizer"]["search_bounds"],
            )
            self._arx_rls_history = {
                "time": [],
                "soc": [],
                "temperature": [],
                "r0": [],
                "r1": [],
                "c1": [],
                "ocv_arx": [],
                "alpha": [],
                "tau1": [],
                "ocv_dt": [],
                "ocv_error": [],
                "ocv_offset_charge": [],
                "ocv_offset_discharge": [],
            }
            self._ocv_offset_charge = 0.0
            self._ocv_offset_discharge = 0.0
            self._ocv_offset_alpha = arx_config.get("ocv_offset_alpha", 0.1)
            self._ocv_offset_limit = arx_config.get("ocv_offset_limit", 0.08)
            self._enable_ocv_offset = arx_config.get("enable_ocv_offset", False)
        else:
            self._arx_rls = None

    def get_domain_vars(self):
        """
        Get the domain variables of the parameter space.
        """
        return self._param_space.domain_variables

    def get_estimated_params(self):
        """
        Get the estimated parameters.
        """
        params = self._param_space.active_region.centroid_dict.copy()

        if self._estimation_method == "arx_rls" and self._enable_ocv_offset:
            params["ocv_offset_charge"] = self._ocv_offset_charge
            params["ocv_offset_discharge"] = self._ocv_offset_discharge

        return params

    def _add_to_batch(self, sample: dict):
        """
        Add the sample to the batch.
        
        Args:
            sample (dict): The sample to add to the batch
        """
        self._input_batch.append(sample)
        
    def _clear_batch(self):
        """
        Clear the batch.
        """
        self._input_batch = []

    def _collect_data_points(self, 
                             points: List[float],                             
                             cluster_name: str,
                             is_outlier: bool
                             ):
        """
        Add data points to the history parameters.
        """
        for point in points:
            for key in point:
                if key not in self._data_history:
                    warnings.warn(f"Key {key} not in data history. Skipping.")
                self._data_history[key].append(point[key])
            self._data_history['cluster'].append(cluster_name)
            self._data_history['is_outlier'].append(is_outlier)
            self._data_history['active_cluster'].append(self._param_space.active_region.name)

    def reset(self, **kwargs):
        """
        Reset the adapter.
        """
        self._input_batch = []
        self._init_state = {}
        self._param_space.select_active_region()    # 'Latest' by default
        self._adaptation_time_step = 0

        if self._arx_rls is not None:
            self._arx_rls.reset()

        for region in self._param_space.regions:
            self._collect_data_points(points=region.cluster.to_dict(orient='records'),
                                      cluster_name=region.name,
                                      is_outlier=False)

    def set_init_state(self, init_state: dict):
        """
        Set the initial state of each optimization loop.
        
        Args:
            init_state (dict): The initial state of the simulation.
        """
        self._init_state = init_state

    def keep_monitoring(self):
        """
        Return True if the adapter is keeping monitoring the parameters.
        """
        return len(self._input_batch) < self._batch_size

    def track_simulation_state(self, sample: dict, domain_vars: dict):
        """
        Track the sample.
        
        Args:
            sample (dict): The sample to track.
        """
        for key in self.get_domain_vars():
            sample[key] = domain_vars[key]

        if self._estimation_method == "arx_rls":
            if "v_oc_lut" in domain_vars:
                sample["v_oc_lut"] = domain_vars["v_oc_lut"]
            elif "v_oc" in domain_vars:
                # Fallback for a model without a dedicated LUT-only key --
                # note this is the model's own previous output (may already
                # include an offset), not a clean reference.
                sample["v_oc_lut"] = domain_vars["v_oc"]

        # With more than one regions, we need to select the active region
        if len(self._param_space.regions) > 1:
            # 'Latest' mode to select the active region (uncomment for 'closest' mode)
            # mean_domain = self._param_space.check_batch_mean_domain(input_batch=self._input_batch, window_size=60)
            # self._param_space.select_active_region(point=mean_domain)
            self._param_space.select_active_region()
        
        # If no active region is selected, select it based on the current sample (can happen with 'closest' mode)
        if self._param_space.active_region is None:
            self._param_space.select_active_region(point={key: sample[key] for key in self.get_domain_vars()})
            
        self._add_to_batch(sample)

    def step(self):
        """
        Enhanced step method with time series visualization support.
        """
        if self._estimation_method == "arx_rls":
            theta = self._arx_rls.estimate_from_batch(self._input_batch)

            if theta is None:
                print(
                    "ARX/RLS produced no valid theta. "
                    f"updates={self._arx_rls.n_updates}, "
                    f"reason={self._arx_rls.last_rejection_reason}"
                )
                self._clear_batch()
                return
        else:
            # Theta is associated to the mean of the domain variables in the batch
            theta = self._optimizer.estimate_new_theta(self._init_state, self._input_batch, centroid=self._param_space.active_region.centroid)

        mean_domain = self._param_space.check_batch_mean_domain(input_batch=self._input_batch)

        if self._estimation_method == "arx_rls":
            full = self._arx_rls.last_valid_full_params

            if full is not None:
                ocv_dt_values = [
                    sample["v_oc_lut"]
                    for sample in self._input_batch
                    if "v_oc_lut" in sample and np.isfinite(sample["v_oc_lut"])
                ]

                if ocv_dt_values:
                    ocv_dt = np.median(ocv_dt_values)
                    raw_offset = full["ocv"] - ocv_dt
                    raw_offset = np.clip(
                        raw_offset,
                        -self._ocv_offset_limit,
                        self._ocv_offset_limit,
                    )

                    # See cluster_shift.py for the sign-convention rationale:
                    # raw current < 0 corresponds to physical discharge under
                    # this experiment's battery.sign_convention: "passive".
                    # Only the offset matching the batch's dominant *raw*
                    # current sign is refreshed; the other regime's offset is
                    # left untouched until data from that regime dominates a
                    # batch again.
                    batch_currents = [
                        sample["current"]
                        for sample in self._input_batch
                        if "current" in sample and np.isfinite(sample["current"])
                    ]
                    dominant_raw_current = np.median(batch_currents) if batch_currents else 0.0

                    if dominant_raw_current < 0:
                        self._ocv_offset_discharge = (
                            (1.0 - self._ocv_offset_alpha) * self._ocv_offset_discharge
                            + self._ocv_offset_alpha * raw_offset
                        )
                    else:
                        self._ocv_offset_charge = (
                            (1.0 - self._ocv_offset_alpha) * self._ocv_offset_charge
                            + self._ocv_offset_alpha * raw_offset
                        )

                ocv_dt = np.median(ocv_dt_values) if ocv_dt_values else np.nan
                ocv_arx = full["ocv"]

                self._arx_rls_history["time"].append(self._adaptation_time_step + 1)
                self._arx_rls_history["soc"].append(mean_domain.get("soc", np.nan))
                self._arx_rls_history["temperature"].append(mean_domain.get("temperature", np.nan))
                self._arx_rls_history["r0"].append(full["r0"])
                self._arx_rls_history["r1"].append(full["r1"])
                self._arx_rls_history["c1"].append(full["c1"])
                self._arx_rls_history["ocv_arx"].append(ocv_arx)
                self._arx_rls_history["alpha"].append(full["alpha"])
                self._arx_rls_history["tau1"].append(full["tau1"])
                self._arx_rls_history["ocv_dt"].append(ocv_dt)
                self._arx_rls_history["ocv_error"].append(ocv_arx - ocv_dt)
                self._arx_rls_history["ocv_offset_charge"].append(self._ocv_offset_charge)
                self._arx_rls_history["ocv_offset_discharge"].append(self._ocv_offset_discharge)

        # Check the affinity of the new theta with the active region
        if len(self._param_space.clusters) > 1:
            self._param_space.select_active_region(point=mean_domain)
        
        # Determine if point is outlier before adding to parameter space
        point_dict = {dim: val for dim, val in zip(self._param_space.param_variables, theta)}
        
        # Add the new theta to the active region or to the outliers. Here mean_domain is used to add the domain variables to the point.
        is_outlier = self._param_space.eval_estimated_params(params=[point_dict], 
                                                             region=self._param_space.active_region,
                                                             mean_domain=mean_domain,
                                                             curr_time=self._adaptation_time_step)

        print("Evaluation estimated parameters:")
        print("--------------------------------")
        print(f"Time {self._adaptation_time_step}: active '{self._param_space.active_region.name}'", 
              f"with centroid {self._param_space.active_region.centroid},"
              f"\nEstimated parameters: {theta},"
              f"\nIs outlier: {is_outlier}")
        print("================================\n")

        # Increment simulation step counter
        self._adaptation_time_step += 1
        self._collect_data_points(points=[{**point_dict, **mean_domain, 'time': self._adaptation_time_step}],
                                  cluster_name=self._param_space.active_region.name if not is_outlier else '',
                                  is_outlier=is_outlier)

        # Analysis of outliers and faulty cluster creation
        if self._enable_adaptation:
            self._build_faulty_cluster()

        self._clear_batch()

    def _build_faulty_cluster(self):
        """
        Generate a new cluster from the outliers.
        """
        print(">>> Checking for new faulty clusters...")
        if self._param_space.outliers is not None and len(self._param_space.outliers) > self._adaptation_options.get('min_number_outliers', 20):
            ks_result = True

            # Perform KS tests between each cluster and the outliers
            for idx, region in enumerate(self._param_space.regions):
                _, p = ks_test(region.param_points, self._param_space.outliers[self._param_space.param_variables].to_numpy())
                
                # If p-value is greater than alpha, we consider the distributions similar (H0 not rejected)
                if p > self._adaptation_options['ks_test'].get('alpha', 0.05):  # not significantly different
                    ks_result = False

            # If all KS tests passed, look for a new cluster
            if ks_result:
                # Use mountain method and MCD to find the core of the new cluster
                outliers = self._param_space.outliers[self._param_space.param_variables].to_numpy()
                times = self._param_space.outliers['time'].to_numpy()
                _, indices, _, _ = trovo_mountain_method(points=outliers, 
                                                         times=times,
                                                         curr_time=self._adaptation_time_step,
                                                         lambda_=self._adaptation_options['mountain_method'].get('lambda', 0.8),
                                                         radius=self._adaptation_options['mountain_method'].get('radius', 0.2),
                                                         eta=self._adaptation_options['mountain_method'].get('eta', 1e-6),
                                                         max_iter=self._adaptation_options['mountain_method'].get('max_iter', 100))
                support = outliers[indices]
            
                if len(support) > len(self._param_space.param_variables):
                    _, _, idx_list = mcd_custom(points=support, max_iter_factor=self._adaptation_options['mcd'].get('max_iter_factor', 10))

                    # If we have enough outliers, create a new cluster
                    if len(idx_list) > self._adaptation_options.get('min_number_outliers', 20):
                        self._param_space.add_cluster(points=self._param_space.outliers.iloc[np.array(indices)[idx_list]],
                                                      curr_time=self._adaptation_time_step,
                                                      name=f"faulty_{self._n_faulty_clusters}")

                        self._n_faulty_clusters += 1
                        # Remove the points from the outliers
                        if self._adaptation_options.get('drop_all_outliers', False):
                            self._param_space._df_outliers = self._param_space._df_outliers.drop(self._param_space._df_outliers.index)
                        else:
                            self._param_space._df_outliers = self._param_space._df_outliers.drop(self._param_space._df_outliers.index[np.array(indices)[idx_list]])
                        
                        print(f">>> New cluster created with {len(idx_list)} points.\n")
                    
                    else:
                        print(">>> Not enough points to create a new cluster after MCD.\n")   
                else:
                    print(">>> Not enough points to create a new cluster after mountain method.\n")
            else:
                print(">>> KS test failed. No new cluster created.\n")
        else:
            print(">>> Not enough outliers to create a new cluster.\n")
            
            
    def export_data_history(self, filepath: str):
        """Export time series data to CSV."""
        if not self._data_history:
            warnings.warn("No data to export.")
            return
        
        df = pd.DataFrame(self._data_history)
        df.to_csv(f"{filepath}/parameter_evolution.csv", index=False)

        for region in self._param_space.regions:
            region.cluster.to_csv(f"{filepath}/cluster_{region.name}.csv", index=False)

        print(f"Data exported to {filepath}\n")

        if self._estimation_method == "arx_rls" and self._arx_rls_history["time"]:
            df_arx = pd.DataFrame(self._arx_rls_history)
            df_arx.to_csv(f"{filepath}/arx_rls_diagnostics.csv", index=False)
            print(f"Data exported to {filepath}/arx_rls_diagnostics.csv")