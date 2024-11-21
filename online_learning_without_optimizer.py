from src.online_learning.battery_adaptation_without_optimizer import BatteryAdaptationWO
from src.online_learning.utils import load_from_yaml, load_cluster
from src.online_learning.cluster import Cluster
import pandas as pd


if __name__ == "__main__":

    df = pd.read_csv("data/ground/experiment_signals/dataset_0_cutted550samples.csv")
    dataset = {'v_real': df['voltage'].values,
               'i_real': df['current'].values,
               't_real': df['temperature'].values,
               'time': df['time']}

    nominal_clusters = dict()
    file_names = ["src/online_learning/change_detection/surrogate_data/phi_0_kmeans.csv",
                  "src/online_learning/change_detection/surrogate_data/phi_1_kmeans.csv",
                  "src/online_learning/change_detection/surrogate_data/phi_2_kmeans.csv",
                  "src/online_learning/change_detection/surrogate_data/phi_3_kmeans.csv"]

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

    # todo: add the hyperparameters of the change detection.
    print('ranges_kmeans are:', battery_settings['ranges'])

    battery_adaptation = BatteryAdaptationWO(battery_setings=battery_settings,
                                             dataset=dataset,
                                             nominal_clusters=nominal_clusters)

    battery_adaptation.run_experiment()
