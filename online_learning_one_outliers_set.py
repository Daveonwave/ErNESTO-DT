from src.online_learning.battery_adaptation_one_outliers_set import BatteryAdaptation
from src.online_learning.utils import load_from_yaml, load_cluster
from src.online_learning.cluster import Cluster
import pandas as pd


if __name__ == "__main__":

    df = pd.read_csv("data/ground/fault_signals/multiplicative_fault_ground.csv")
    dataset = {'v_real': df['voltage'].values,
               'i_real': df['current'].values,
               't_real': df['temperature'].values,
               'time': df['time']}

    nominal_clusters = dict()
    file_names = ["data/ground/nominal_cluster_kmeans/phi_0_kmeans.csv",
                  "data/ground/nominal_cluster_kmeans/phi_1_kmeans.csv",
                  "data/ground/nominal_cluster_kmeans/phi_2_kmeans.csv",
                  "data/ground/nominal_cluster_kmeans/phi_3_kmeans.csv"]

    for i, file_name in enumerate(file_names):
        nominal_clusters[i] = Cluster()
        nominal_clusters[i].data_points = load_cluster(file_name)
        nominal_clusters[i].compute_covariance()
        nominal_clusters[i].compute_centroid()

    ranges = load_from_yaml('data/external/ranges_k_means')
    new_clusters = {}
    for key, value in ranges.items():
        # Extract the number from the key (e.g., 'cluster_0' -> 0)
        new_key = int(key.split('_')[1])
        new_clusters[new_key] = value

    battery_settings = {'electrical_params': load_from_yaml('data/external/electrical_params'),
                        'thermal_params': load_from_yaml('data/external/thermal_params'),
                        'battery_options': load_from_yaml('data/external/battery_options'),
                        'ranges': new_clusters,
                        'load_var': 'current'}

    # todo: alpha also to be tuned

    optimizer_settings = {'alpha': 0.00, 'batch_size': 40000,
                          'optimizer_method': 'BFGS',
                          'save_results': True, 'number_of_restarts': 1,
                          'bounds': [(0.013612, 0.034656), (0.013954, 0.07027), (976.466, 15552.53)],
                          # RMK: bounds are used to generate randomly the initial guess for the gradient descent
                          'scale_factors': [1e-6, 1e-6, 1],  # useless since the optimizer doesn't exploit it
                          'options': {
                                        'disp': True,
                                        'gtol': 1e-8,
                                        'eps': 1e-10,
                                        'maxiter': 1000}
                          }

    # todo: add here also the hyperparameters of the change detection.

    battery_adaptation = BatteryAdaptation(optimizer_settings=optimizer_settings,
                                           battery_setings=battery_settings,
                                           dataset=dataset,
                                           nominal_clusters=nominal_clusters)

    battery_adaptation.run_experiment()
