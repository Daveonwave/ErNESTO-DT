import yaml
import pandas as pd
import numpy as np
import os
import csv

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


def save_to_csv(data, filename, headers):
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        if isinstance(data, list):
            for item in data:
                writer.writerow([item])
        elif isinstance(data, dict):
            for key, value in data.items():
                writer.writerow([key, value])


def convert_to_dict_list(np_array_list):
    keys = ['r0', 'r1', 'c']

    dict_list = []
    for array in np_array_list:
        flattened_array = array.flatten()
        dict_item = {keys[i]: flattened_array[i] for i in range(len(keys))}
        dict_list.append(dict_item)

    return dict_list


def save_dict_list_to_csv(dict_list, filename):
    df = pd.DataFrame(dict_list)
    df.to_csv(filename, index=False)
