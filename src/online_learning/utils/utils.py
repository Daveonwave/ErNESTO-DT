import yaml
import os

def load_from_yaml(file_path):
    with open(file_path, 'r') as file:
         return yaml.safe_load(file)

def get_absolute_path(relative_path):
    current_dir = os.getcwd()
    absolute_path = os.path.abspath(os.path.join(current_dir, relative_path))
    return absolute_path