from src.online_learning.battery_adaptation import BatteryAdaptation
from src.online_learning.utils import load_from_yaml, load_cluster
from src.online_learning.cluster import Cluster
import pandas as pd


if __name__ == "__main__":

    df = pd.read_csv("data/ground/experiment_signals/dataset_0_cuttedX10.csv")
    dataset = {'v_real': df['voltage'].values,
               'i_real': df['current'].values,
               't_real': df['temperature'].values,
               'time': df['time']}

    nominal_clusters = dict()
    file_names = ["data/ground/nominal_cluster/phi_one.csv",
                  "data/ground/nominal_cluster/phi_two.csv",
                  "data/ground/nominal_cluster/phi_three.csv",
                  "data/ground/nominal_cluster/phi_four.csv"]

    for i, file_name in enumerate(file_names):
        nominal_clusters[i] = Cluster()
        nominal_clusters[i].data_points = load_cluster(file_name)
        nominal_clusters[i].compute_covariance()
        nominal_clusters[i].compute_centroid()

    battery_settings = {'electrical_params': load_from_yaml('data/external/electrical_params'),
                        'thermal_params': load_from_yaml('data/external/thermal_params'),
                        'battery_options': load_from_yaml('data/external/battery_options'),
                        'ranges': load_from_yaml('data/external/ranges'),
                        'load_var': 'current'}

    optimizer_settings = {'alpha': 0.5, 'batch_size': 10000,
                          'optimizer_method': 'L-BFGS-B',
                          'save_results': True, 'number_of_restarts': 1,
                          'bounds': [(0, 1000), (0, 1000), (0, 1000)],
                          'scale_factors': [1e-4, 1e-4, 1e3]}

    # todo: add the hyperparameters of the change detection.

    battery_adaptation = BatteryAdaptation(optimizer_settings=optimizer_settings,
                                           battery_setings=battery_settings,
                                           dataset=dataset,
                                           nominal_clusters=nominal_clusters)

    battery_adaptation.run_experiment()
