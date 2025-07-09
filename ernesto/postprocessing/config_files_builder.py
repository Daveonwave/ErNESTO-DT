import yaml
from pathlib import Path
from ernesto.preprocessing.schema import read_yaml, _check_schema
import os


def build_config_yaml(params:dict, 
                      old_params_file: str, 
                      dest:Path, 
                      filename:str):
    """
    Build a yaml configuration file from a dictionary and an old configuration file.
    
    Args:
        params (dict): dictionary with the configuration parameters.
        old_params_file (str): path to the old configuration file.
        dest (Path): destination folder to save the file.
        filename (str): name of the file.
    """
    try:
        old_params = read_yaml(old_params_file, yaml_type='driven')
    except Exception as e:
        print(f"Error reading the old config file: {e}")
        exit(1)
    
    for param in params:
        # Change initial configuration of the battery such as {soc, soh, initital voltage, current and temp}
        if param in old_params['battery']['init']:
            old_params['battery']['init'][param] = params[param]
    
    os.makedirs(dest, exist_ok=True)
    new_file = dest / filename
    
    with open(new_file, 'w') as file:
        yaml.dump(old_params, file)
        
    try:
        _check_schema(old_params, 'driven')
    except Exception as e:
        print(f"Error reading the new electrical model file: {e}")
        exit(1)
        
    print("New configuration file created successfully!")
        
        
def build_electrical_yaml(params:dict, 
                          old_params_file: str, 
                          dest:Path, 
                          filename:str
                          ):
    """
    Build a yaml configuration file from a dictionary and old electrical model file.
    
    Args:
        params (dict): dictionary with the configuration parameters.
        old_params_file (str): path to the old configuration file.
        dest (Path): destination folder to save the file.
        filename (str): name of the file.
    """
    try:
        old_params = read_yaml(old_params_file, yaml_type='model')
    except Exception as e:
        print(f"Error reading the old electrical model file: {e}")
        exit(1)
        
    for param in params:
        # Change electrical parameters of the battery such as R0, R1, and C1.
        # This happens only if the parameter is scalar and not a lookup table.
        if param in old_params['components']:
            if old_params['components'][param]['selected_type'] == 'scalar':
                old_params['components'][param]['scalar'] = params[param]
    
    os.makedirs(dest, exist_ok=True)
    new_file = dest / filename
        
    with open(new_file, 'w') as file:
        yaml.dump(old_params, file)
        
    try:
        _check_schema(old_params, 'model')
    except Exception as e:
        print(f"Error reading the new electrical model file: {e}")
        exit(1)
        
    print("New configuration file created successfully!")
        
        
if __name__ == "__main__":
        
    new_params = {'soc': 0.5, 
                  'soh': 0.9, 
                  'voltage': 3.7, 
                  'current': 0.0, 
                  'temperature': 25.0, 
                  'r0': 100,
                  'r1': 0.0000001,
                  'c': 10000000}
    

    build_config_yaml(params=new_params, 
                      old_params_file='./data/config/sim_without_ground.yaml', 
                      dest=Path('./config_files/'), 
                      filename='new_config.yaml')
    
    build_electrical_yaml(params=new_params,
                          old_params_file='./data/config/models/electrical/thevenin_enrica.yaml',
                          dest=Path('./config_files/'),
                          filename='new_electrical.yaml')