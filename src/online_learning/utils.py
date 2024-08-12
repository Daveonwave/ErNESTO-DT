import yaml
import pandas as pd
import numpy as np
import os

def load_from_yaml(file_path):
    with open(file_path, 'r') as file:
         return yaml.safe_load(file)

def get_absolute_path(relative_path):
    current_dir = os.getcwd()
    absolute_path = os.path.abspath(os.path.join(current_dir, relative_path))
    return absolute_path

def load_cluster(csv_string):
    data = pd.read_csv(csv_string)
    array_elements = data[['R_0', 'R_1', 'C_1']].to_numpy()
    list_of_arrays = [np.array(row) for row in array_elements]
    return list_of_arrays